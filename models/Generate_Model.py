import torch
from torch import nn
from models.Temporal_Model import *
import torchaudio
import math
from AudioMAE import audio_models_vit
from timm.models.layers import to_2tuple
from timm.models import create_model
from models import models_vit
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
from timm.layers import PatchEmbed, Mlp,LayerType
# from timm.models.vision_transformer import PatchEmbed, Mlp,LayerType
from models.adapter import Block
from functools import partial
import os
from models.DSCMT import CA ,SA

def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
        gs_old = None,
) -> torch.Tensor:
    # function from timm
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    if gs_old is None:
        gs_old = (int(math.sqrt(len(posemb_grid))), int(math.sqrt(len(posemb_grid))))

    if gs_new is None or not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

class PatchEmbed_new(nn.Module):
    '''
    copied from AudioMAE
    '''
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape

    def forward(self, x):
        # print('shape: ', x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GenerateModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


        self.CMTM = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.our_classifier = torch.nn.Linear(512,args.number_class)
        self.vision_proj = torch.nn.Linear(768,512)
        # self.video_proj = torch.nn.Linear(768, 512)
        # self.audio_proj = torch.nn.Linear(768, 512)

        # self.n_audio = 256
        self.n_audio = 196

        self.n_image = (args.img_size // 16 )**2
        self.n_progr = 3
        self.test_ca =  nn.ModuleList([CA(d_model=768, nhead=8) for _ in range(1)])
        self._build_image_model(img_size=args.img_size)
        self._build_audio_model()

        args.n_fusion_layers = 2



    def _build_audio_model(self, model_name='vit_base_patch16', drop_path_rate=0.1, global_pool=False, mask_2d=True, use_custom_patch=False, ckpt_path='/home/yangbohui/BFF-DFER/models/audiomae_pretrained.pth'):
        self.audio_model = audio_models_vit.__dict__[model_name](
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            mask_2d=mask_2d,
            use_custom_patch=use_custom_patch,
            n_seq = self.n_audio,
            n_progr = self.n_progr)

        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_ratio: float = 4.0
        qkv_bias: bool = True
        qk_norm: bool = False
        init_values: Optional[float] = None
        proj_drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer: Optional[LayerType] = None
        act_layer: Optional[LayerType] = None
        block_fn: Type[nn.Module] = Block
        mlp_layer: Type[nn.Module] = Mlp
        mlp_ratio = 4
        qkv_bias = True
        norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.audioblocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])

        print("self.blocks", self.audioblocks)

        self.audio_model.blocks = self.audioblocks


        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = ckpt['model']
        orig_pos_embed =  ckpt['pos_embed'] # torch.Size([1, 513, 768])
        print(orig_pos_embed.shape, self.audio_model.pos_embed.shape) #torch.Size([1, 513, 768]) torch.Size([1, 197, 768])
        new_posemb = resize_pos_embed(orig_pos_embed, self.audio_model.pos_embed, gs_old=(1024//16,128//16), gs_new=(448//16,112//16)) # use PyTorch function linked above
        ckpt['pos_embed'] = new_posemb

        emb =  torch.randn(1, self.n_audio  + 1, 768)
        emb[:,:self.n_audio+1] = ckpt['pos_embed'][:,:self.n_audio+1]
        del ckpt['pos_embed'] #= emb
        self.audio_model.patch_embed = PatchEmbed_new(img_size=(448,112), patch_size=(16,16), in_chans=1, embed_dim=768, stride=16) # no overlap. stride=img_size=16
        self.audio_model.pos_embed = nn.Parameter(emb, requires_grad=False) # setting to true from outside
        msg = self.audio_model.load_state_dict(ckpt, strict=False)
        print('Audio checkpoint loading: ', msg)


    def _build_image_model(self, model_name='vit_base_patch16', ckpt_path='/home/yangbohui/BFF-DFER/models/mae_face_pretrain_vit_base.pth',
                           global_pool = False, num_heads=12, drop_path_rate=0.1, img_size=224, n_frames=16):

        self.image_encoder = getattr(models_vit, model_name)(
            global_pool=global_pool,
            num_classes=num_heads,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            n_seq = self.n_image,
            n_progr = self.n_progr,
            n_frames=n_frames,
            )

        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_ratio: float = 4.0
        qkv_bias: bool = True
        qk_norm: bool = False
        init_values: Optional[float] = None
        proj_drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer: Optional[LayerType] = None
        act_layer: Optional[LayerType] = None
        block_fn: Type[nn.Module] = Block
        mlp_layer: Type[nn.Module] = Mlp
        mlp_ratio = 4
        qkv_bias = True
        norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                mlp_layer=mlp_layer,
                use_adapter=True
            )
            for i in range(depth)])

        print("self.blocks", self.blocks)

        self.image_encoder.blocks = self.blocks

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        orig_pos_embed =  checkpoint_model['pos_embed'] # torch.Size([1, 197, 768])
        new_posemb = resize_pos_embed(orig_pos_embed, self.image_encoder.pos_embed) # use PyTorch function linked above
        checkpoint_model['pos_embed'] = new_posemb

        msg = self.image_encoder.load_state_dict(checkpoint_model, strict=False)
        print('Image checkpoint loading: ', msg)


        pos_embed = torch.randn(1, self.image_encoder.pos_embed.size(1) , 768)   #torch.Size([1, 203, 768])
        pos_embed[:,:,:] = self.image_encoder.pos_embed

        self.image_encoder.pos_embed = nn.Parameter(pos_embed)  # torch.Size([1, 203, 768])



    def fusion_adapter_simple(self,ii, v, a):

        # layer_dropout = 0.2 * (1 - ii / 12)  # Decrease dropout for deeper layers
        # self.dropout = nn.Dropout(layer_dropout)

        B, L, C = v.shape

        shortcut_v = v
        shortcut_a = a

        # MSA MHSA Stage
        norm_v = self.image_encoder.blocks[ii].norm1(v)
        norm_a = self.audio_model.blocks[ii].norm1(a)

        attn_v = self.image_encoder.blocks[ii].attn(norm_v)
        attn_a = self.audio_model.blocks[ii].attn(norm_a)

        attn_v = self.image_encoder.blocks[ii].drop_path1(attn_v)
        attn_a = self.audio_model.blocks[ii].drop_path1(attn_a)

        # S_Adapter
        vs_hidden = self.image_encoder.blocks[ii].S_Adapter.non_linear_func(self.image_encoder.blocks[ii].S_Adapter.down_proj(norm_v))  # [n, bt, d]
        as_hidden = self.image_encoder.blocks[ii].S_Adapter.non_linear_func(self.image_encoder.blocks[ii].S_Adapter.down_proj(norm_a))
        vs_hidden = self.image_encoder.blocks[ii].S_Adapter.up_proj(vs_hidden)
        as_hidden = self.image_encoder.blocks[ii].S_Adapter.up_proj(as_hidden)


        attn_v = attn_v +  vs_hidden
        attn_a = attn_a +  as_hidden


        v = shortcut_v + attn_v
        a = shortcut_a + attn_a

        # # MSA MLP Stage
        norm_vf = self.image_encoder.blocks[ii].norm2(v)
        norm_af = self.audio_model.blocks[ii].norm2(a)

        vn = self.image_encoder.blocks[ii].mlp(norm_vf)
        an = self.audio_model.blocks[ii].mlp(norm_af)


        # G_Adapter
        vn_hidden = self.image_encoder.blocks[ii].G_Adapter.non_linear_func(self.image_encoder.blocks[ii].G_Adapter.down_proj(norm_vf))
        an_hidden = self.image_encoder.blocks[ii].G_Adapter.non_linear_func(self.image_encoder.blocks[ii].G_Adapter.down_proj(norm_af))


        vn_hidden = self.image_encoder.blocks[ii].G_Adapter.up_proj(vn_hidden)
        an_hidden = self.image_encoder.blocks[ii].G_Adapter.up_proj(an_hidden)


        vn = self.image_encoder.blocks[ii].drop_path2(vn)
        an = self.audio_model.blocks[ii].drop_path2(an)

        v = v + vn + vn_hidden  # self.drop_path(0.5 * )
        a = a + an + an_hidden

        # BFA Stage

        # BFA Fisrt Stage
        norm_vn = self.image_encoder.blocks[ii].norm2(v)
        norm_an = self.audio_model.blocks[ii].norm2(a)

        vn_hidden = self.image_encoder.blocks[ii].Adapter1.non_linear_func(
            self.image_encoder.blocks[ii].Adapter1.down_proj(norm_vn))
        an_hidden = self.audio_model.blocks[ii].Adapter1.non_linear_func(
            self.audio_model.blocks[ii].Adapter1.down_proj(norm_an))

        vn_hidden = self.image_encoder.blocks[ii].Adapter1.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].Adapter1.up_proj(an_hidden)

        # vn_hidden = self.dropout(vn_hidden)
        # an_hidden = self.dropout(an_hidden)


        # v = v  + torch.sigmoid(self.image_encoder.blocks[ii].gate_fusion1) * vn_hidden # self.drop_path(0.5 * )
        # a = a  + torch.sigmoid(self.audio_model.blocks[ii].gate_fusion1)  * an_hidden  # self.drop_path(0.5 *)
        v = v  + nn.functional.tanh(self.image_encoder.blocks[ii].gate_fusion1) * vn_hidden # self.drop_path(0.5 * )
        a = a  + nn.functional.tanh(self.audio_model.blocks[ii].gate_fusion1)  * an_hidden  # self.drop_path(0.5 *)


        # BFA Second Stage
        norm_vn = self.image_encoder.blocks[ii].norm2(v)
        norm_an = self.audio_model.blocks[ii].norm2(a)
        vn_hidden = self.image_encoder.blocks[ii].Adapter2.non_linear_func(
            self.image_encoder.blocks[ii].Adapter2.down_proj(norm_vn))
        an_hidden = self.audio_model.blocks[ii].Adapter2.non_linear_func(
            self.audio_model.blocks[ii].Adapter2.down_proj(norm_an))

        vn_hidden = self.image_encoder.blocks[ii].Adapter2.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].Adapter2.up_proj(an_hidden)

        # vn_hidden = self.dropout(vn_hidden)
        # an_hidden = self.dropout(an_hidden)

        # v = v  + torch.sigmoid(self.image_encoder.blocks[ii].gate_fusion2) * an_hidden.repeat_interleave(16,0)# self.drop_path(0.5 * )
        # a = a  + torch.sigmoid(self.audio_model.blocks[ii].gate_fusion2)  * vn_hidden.view(a.shape[0], 16, -1, 768).mean(1)  # self.drop_path(0.5 *)

        v = v  + nn.functional.tanh(self.image_encoder.blocks[ii].gate_fusion2) * an_hidden.repeat_interleave(16,0)# self.drop_path(0.5 * )
        a = a  + nn.functional.tanh(self.audio_model.blocks[ii].gate_fusion2)  * vn_hidden.view(a.shape[0], 16, -1, 768).mean(1)  # self.drop_path(0.5 *)


        return v,a

    def forward(self, image, audio):

        n, t, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        assert t == 16
        B = image.shape[0]
        print("Input device:", image.device, audio.device)
        for ii in range(len(self.audio_model.blocks) ):

            audio = self.audio_model.forward_block_pre(ii, audio)
            image = self.image_encoder.forward_block_pre(ii, image,
                                                         B)


            image, audio = self.fusion_adapter_simple(ii, image, audio)

            image, origin_image = self.image_encoder.forward_block_post(ii, image, B)
            audio, origin_audio = self.audio_model.forward_block_post(ii, audio)


        # CMTM Stage
        image = image.contiguous().view(n, t, -1)
        fuse_cls = torch.cat((audio.unsqueeze(1), image), dim=1)
        fuse_cls = self.vision_proj(fuse_cls)
        v_output = self.CMTM(fuse_cls)
        output = self.our_classifier(v_output)

        return output

