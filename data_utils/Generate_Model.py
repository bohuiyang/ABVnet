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
from testblock import Block
from functools import partial
import os
# import pathlib
# from pathlib import PosixPath, WindowsPath
# pathlib.PosixPath = pathlib.WindowsPath
from models.DSCMT import CA ,SA # 确保路径正确


class MLP_adapter(nn.Module):
    # Non-Linear Transformation in the paper, acting as the translator between modalities.
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


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
        print('shape: ', x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # 获取序列长度
        n = x.shape[2] - 1  # 序列长度减一

        # 计算每个位置上的像素值与该通道均 值的差的平方
        d = (x - x.mean(dim=2, keepdim=True)).pow(2)

        # 求得每个通道的方差
        v = d.sum(dim=2, keepdim=True) / n

        # 计算重要性权重
        E_inv = d / (4 * (v + self.lambda_val)) + 0.5

        # 返回加权后的特征
        return x * torch.sigmoid(E_inv)

class GenerateModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 初始化时序Transformer分类器，用于处理视频帧序列的时序信息
        # 参数解释：
        # - num_patches: 每帧图像划分为的patch数量，这里假设每帧图像大小被划分为4x4的patch，因此总共是16个patch
        # - input_dim: 每个patch的输入维度，经过先前的处理，每个patch的特征维度为512
        # - depth: Transformer编码器的层数，通过命令行参数指定，提供网络深度的灵活性
        # - heads: 多头自注意力机制的头数，这里设置为8，意味着在自注意力计算中并行运行8个不同的注意力机制
        # - mlp_dim: 前馈神经网络（MLP）的维度，在自注意力计算之后用于对特征进行非线性变换，此处设置为1024
        # - dim_head: 每个注意力头的维度，这里是64，控制每个注意力机制的输出维度
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.our_classifier = torch.nn.Linear(512,args.number_class)
        self.vision_proj = torch.nn.Linear(768,512)

        # self.n_audio = 256
        self.n_audio = 196

        self.n_image = (args.img_size // 16 )**2
        self.n_progr = 3

        self._build_image_model(img_size=args.img_size)
        self._build_audio_model()
        self.w = nn.Parameter(torch.ones(2))  # 初始化权重, 对2个卷积分别加一个权重

        # self.sa1 = SA(768, 8, 2048, 0.1)
        # self.sa2 = SA(768, 8, 2048, 0.1)
# 21335MiB / 24576MiB  21653MiB / 24576MiB  23887MiB / 24576MiB 23599MiB / 24576MiB   20837MiB / 24576MiB
        self.ca_list = nn.ModuleList([CA(d_model=768, nhead=8)
                                      for _ in range(1)])  # 1  21291MiB / 24576MiB
        # self.ca = CA(d_model=768, nhead=8)

        self.norm_layer = nn.LayerNorm(384)
        self.norm_layer_late = nn.LayerNorm(128)

        self.gate_v = nn.Parameter(torch.zeros(1))
        self.gate_a = nn.Parameter(torch.zeros(1))

        self.conv1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1)

        args.n_qp = 4
        args.n_fcp = 4
        args.n_qcp = 4
        args.n_fusion_layers = 2
        self.v2t_qp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, args.n_qp, 768).normal_(std=0.02)) for _ in
             range(args.n_fusion_layers)])
        self.t2v_qp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, args.n_qp, 768).normal_(std=0.02)) for _ in
             range(args.n_fusion_layers)])

        self.v2t_qcp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, args.n_qcp, 768).normal_(std=0.02)) for _ in
             range(args.n_fusion_layers)])
        self.t2v_qcp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, args.n_qcp, 768).normal_(std=0.02)) for _ in
             range(args.n_fusion_layers)])

        if args.n_fcp > 0:
            self.vision_fcp = nn.ParameterList(
                [nn.Parameter(torch.empty(1, args.n_fcp, 768).normal_(std=0.02)) for _
                 in range(args.n_fusion_layers)])
            self.text_fcp = nn.ParameterList(
                [nn.Parameter(torch.empty(1, args.n_fcp,768).normal_(std=0.02)) for _
                 in range(args.n_fusion_layers)])

        self.v2t_trans = nn.ModuleList(
            [MLP_adapter(768, 384, 768)
             for _ in range(args.n_fusion_layers)])
        self.t2v_trans = nn.ModuleList(
            [MLP_adapter(768, 384, 768)
             for _ in range(args.n_fusion_layers)])



    def _build_audio_model(self, model_name='vit_base_patch16', drop_path_rate=0.1, global_pool=False, mask_2d=True, use_custom_patch=False, ckpt_path='models/audiomae_pretrained.pth'):
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


    def _build_image_model(self, model_name='vit_base_patch16', ckpt_path='models/mae_face_pretrain_vit_base.pth',
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
        # self.args.modelname = 'vit_base_patch16_224'
        # self.args.nb_classes = 7
        # self.args.num_frames = 16
        # self.args.num_segments = 1
        # self.args.tubelet_size = 2
        # self.args.drop = 0.0
        # self.args.drop_path = 0.0
        # self.args.attn_drop_rate = 0.0
        # self.args.use_mean_pooling = True
        # self.args.init_scale = 0.0001
        # self.testmodel = create_model(
        #     self.args.modelname,
        #     pretrained=False,
        #     num_classes=self.args.nb_classes,
        #     all_frames=self.args.num_frames * self.args.num_segments,
        #     tubelet_size=self.args.tubelet_size,
        #     drop_rate=self.args.drop,
        #     drop_path_rate=self.args.drop_path,
        #     attn_drop_rate=self.args.attn_drop_rate,
        #     drop_block_rate=None,
        #     use_mean_pooling=self.args.use_mean_pooling,
        #     init_scale=self.args.init_scale,
        #     tuning_config=self.args.tuning_config,
        # )
        # patch_size = self.testmodel.patch_embed.patch_size
        # print("Patch size = %s" % str(patch_size))
        # # self.args.window_size = (self.args.num_frames // 2, self.args.input_size // patch_size[0], self.args.input_size // patch_size[1])
        # # self.args.patch_size = patch_size
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

        # # freeze all but the head
        # for name, p in self.image_encoder.named_parameters():
        #     if name in msg.missing_keys:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False if not args.fulltune else True
        # for _, p in self.image_encoder.head.named_parameters():
        #     p.requires_grad = True
        # for _, p in self.image_encoder.fc_norm.named_parameters():
        #     p.requires_grad = True

        pos_embed = torch.randn(1, self.image_encoder.pos_embed.size(1) , 768)   #torch.Size([1, 203, 768])
        pos_embed[:,:,:] = self.image_encoder.pos_embed

        self.image_encoder.pos_embed = nn.Parameter(pos_embed)  # torch.Size([1, 203, 768])
        self.use_act = True

    def fusion_adapter(self,ii, v, a):



        B, L, C = v.shape
        # v = rearrange(v, '(b t) n c -> (b n) t c', t=16, n=L, c=C)
        # res_temporal = self.image_encoder.blocks[ii].attn(self.image_encoder.blocks[ii].norm1(v))
        # res_temporal = self.image_encoder.blocks[ii].T_Adapter(res_temporal, add_residual=False)
        # v = v + self.image_encoder.blocks[ii].drop_path1(res_temporal)
        # v = rearrange(v, '(b n) t c -> (b t) n c', t=16, n=L, c=C)

        # a = rearrange(a, '(b t) n c -> (b n) t c', t=1, n=L, c=C)
        # res_temporal_a = self.audio_model.blocks[ii].attn(self.audio_model.blocks[ii].norm1(a))
        # res_temporal_a = self.audio_model.blocks[ii].T_Adapter(res_temporal_a)
        # a = a + self.audio_model.blocks[ii].drop_path1(res_temporal_a)
        # a = rearrange(a, '(b n) t c -> (b t) n c', t=1, n=L, c=C)

        # self.drop_path = nn.Dropout(0.1)

        # 简化 adapter

        shortcut_v = v
        # x = self.norm1(v)

        shortcut_a = a
        # a = self.norm1(a)

        # attn_v = self.image_encoder.blocks[ii].attn(self.image_encoder.blocks[ii].norm1(v))
        # attn_a = self.audio_model.blocks[ii].attn(self.audio_model.blocks[ii].norm1(a))
        norm_v = self.image_encoder.blocks[ii].norm1(v)
        norm_a = self.audio_model.blocks[ii].norm1(a)

        attn_v = self.image_encoder.blocks[ii].attn(norm_v)
        attn_a = self.audio_model.blocks[ii].attn(norm_a)

        attn_v = self.image_encoder.blocks[ii].drop_path1(attn_v)
        attn_a = self.audio_model.blocks[ii].drop_path1(attn_a)

        if self.use_act== True :
            vs_hidden = self.image_encoder.blocks[ii].SAdapter2.non_linear_func(self.image_encoder.blocks[ii].SAdapter2.down_proj(norm_v))  # [n, bt, d]
            as_hidden = self.audio_model.blocks[ii].SAdapter2.non_linear_func(self.audio_model.blocks[ii].SAdapter2.down_proj(norm_a))
        else:
            vs_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter2.down_proj(norm_v))  # [n, bt, d]
            as_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter2.down_proj(norm_a))

        # # 加入layernorm尝试
        # vs_hidden = self.norm_layer(vs_hidden).to(vs_hidden.device)
        # as_hidden = self.norm_layer(as_hidden).to(as_hidden.device)

        vs_fuse = vs_hidden  # [bt, nv, d]
        as_fuse = as_hidden  # [bt, na, d]

        as_fuse  = as_fuse.repeat_interleave(16,0)
        # .view(n, t, -1, 768).mean(dim=1)
        attn_vs = F.softmax(torch.bmm(vs_fuse, as_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
        a2v_res_s = torch.bmm(attn_vs, as_fuse)  # [bt, nv, d]

        attn_as = F.softmax(torch.bmm(as_fuse, vs_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
        v2a_res_s = torch.bmm(attn_as, vs_fuse)  # [bt, na, d]

        vs_hidden = vs_hidden + self.image_encoder.blocks[ii].gate * a2v_res_s
        as_hidden = as_hidden + self.audio_model.blocks[ii].gate * v2a_res_s.view(a.shape[0], 16, -1, 128).mean(1)
        # vs_hidden = vs_hidden +  a2v_res_s.mean(1).unsqueeze(1)
        # as_hidden = as_hidden +  v2a_res_s.view(a.shape[0], 16, -1, 128).mean(1).mean(1).unsqueeze(1)
        print("ii:",ii,"   self.image_encoder.blocks[ii].gate:",self.image_encoder.blocks[ii].gate)



        # attn_v = attn_v + self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden)
        # attn_a = attn_a + self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden)

        vs_hidden = self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden)
        as_hidden = self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden)

        # attn_v = attn_v + self.image_encoder.blocks[ii].gate * as_hidden.repeat_interleave(16,0)
        # attn_a = attn_a + self.audio_model.blocks[ii].gate * vs_hidden.view(a.shape[0], 16, -1, 768).mean(1)
        attn_v = attn_v +  vs_hidden
        attn_a = attn_a +  as_hidden

        # attn_v = attn_v + self.image_encoder.blocks[ii].drop_path1(self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden))
        # attn_a = attn_a + self.audio_model.blocks[ii].drop_path1(self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden))


        # v = v.view(B_v, H * W, C)
        # x = shortcut + self.drop_path(x)
        # v = shortcut_v + self.image_encoder.blocks[ii].drop_path1(attn_v)
        v = shortcut_v + attn_v

        # a = a.view(B_a, H * W, C)
        # x = shortcut + self.drop_path(x)
        # a = shortcut_a + self.audio_model.blocks[ii].drop_path1(attn_a)
        a = shortcut_a + attn_a


        # FFN
        norm_vf = self.image_encoder.blocks[ii].norm2(v)
        norm_af = self.audio_model.blocks[ii].norm2(a)

        vn = self.image_encoder.blocks[ii].mlp(norm_vf)
        an = self.audio_model.blocks[ii].mlp(norm_af)

        # vn = self.image_encoder.blocks[ii].drop_path2(vn)
        # an = self.audio_model.blocks[ii].drop_path2(an)

        if self.use_act == True:
            vn_hidden = self.image_encoder.blocks[ii].adaptmlp.non_linear_func(self.image_encoder.blocks[ii].adaptmlp.down_proj(norm_vf))
            an_hidden = self.audio_model.blocks[ii].adaptmlp.non_linear_func(self.audio_model.blocks[ii].adaptmlp.down_proj(norm_af))
        else :
            vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].adaptmlp.down_proj(norm_vf))
            an_hidden = self.norm_layer(self.audio_model.blocks[ii].adaptmlp.down_proj(norm_af))

        # # 加入layernorm尝试
        # vn_hidden = self.norm_layer_late(vn_hidden).to(vn_hidden.device)
        # an_hidden = self.norm_layer_late(an_hidden).to(an_hidden.device)

        vn_fuse = vn_hidden
        an_fuse = an_hidden

        an_fuse = an_fuse.repeat_interleave(16, 0)
        attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
        a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]

        attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
        v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]

        # vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
        # an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 64).mean(dim=1)
        vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
        an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)
        # vn_hidden = vn_hidden +  a2v_res_n.mean(1).unsqueeze(1)
        # an_hidden = an_hidden + v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)

        vn_hidden = self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)
        # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
        # self.drop_path = nn.Dropout(0.1)

        vn = self.image_encoder.blocks[ii].drop_path2(vn)
        an = self.audio_model.blocks[ii].drop_path2(an)
        # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
        # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)

        # v = v + vn + self.image_encoder.blocks[ii].gate_fusion * an_hidden.repeat_interleave(16,0) # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].gate_fusion  * vn_hidden.view(a.shape[0], 16, -1, 768).mean(1)  # self.drop_path(0.5 *)
        v = v + vn + self.image_encoder.blocks[ii].drop_path2 (vn_hidden)   # self.drop_path(0.5 * )
        a = a + an + self.audio_model.blocks[ii].drop_path2(an_hidden)
              # self.drop_path(0.5 *)


        #feed 之前     feed后  上采样后
        # feed 后       上采样后


        # 在融合

        # FFN
        norm_vn = self.image_encoder.blocks[ii].norm2(v)
        norm_an = self.audio_model.blocks[ii].norm2(a)


        # vn = self.image_encoder.blocks[ii].drop_path2(vn)
        # an = self.audio_model.blocks[ii].drop_path2(an)

        if self.use_act == True:
            vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.non_linear_func(
                self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
            an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.non_linear_func(
                self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))
        else:
            vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
            an_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))

        # # 加入layernorm尝试
        # vn_hidden = self.norm_layer_late(vn_hidden).to(vn_hidden.device)
        # an_hidden = self.norm_layer_late(an_hidden).to(an_hidden.device)

        vn_fuse = vn_hidden
        an_fuse = an_hidden

        an_fuse = an_fuse.repeat_interleave(16, 0)
        attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
        a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]

        attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
        v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]

        # vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
        # an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 64).mean(dim=1)
        vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
        an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)
        # vn_hidden = vn_hidden +  a2v_res_n.mean(1).unsqueeze(1)
        # an_hidden = an_hidden + v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)

        vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.up_proj(an_hidden)
        # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
        # self.drop_path = nn.Dropout(0.1)


        # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
        # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)

        v = v  + self.image_encoder.blocks[ii].gate_fusion * vn_hidden # self.drop_path(0.5 * )
        a = a  + self.audio_model.blocks[ii].gate_fusion  * an_hidden  # self.drop_path(0.5 *)

        #按照图示 加的是 add之前的 所以是vn an
        # v = vn + self.image_encoder.blocks[ii].gate_fusion * vn_hidden  # self.drop_path(0.5 * )
        # a = an + self.audio_model.blocks[ii].gate_fusion * an_hidden  # self.drop_path(0.5 *)


        # v = v + vn + self.image_encoder.blocks[ii].drop_path2(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].drop_path2(an_hidden)

        #
        # v = v + norm_vf
        # a = a + norm_af
        # v = self.image_encoder.blocks[ii].norm2(v)
        # a = self.audio_model.blocks[ii].norm2(a)

        if ii < 5 :
            #  尝试stack
            # FFN
            norm_vn = self.image_encoder.blocks[ii].norm2(v)
            norm_an = self.audio_model.blocks[ii].norm2(a)

            # vn = self.image_encoder.blocks[ii].drop_path2(vn)
            # an = self.audio_model.blocks[ii].drop_path2(an)

            if self.use_act == True:
                vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.non_linear_func(
                    self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
                an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.non_linear_func(
                    self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))
            else:
                vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
                an_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))

            vn_fuse = vn_hidden
            an_fuse = an_hidden

            an_fuse = an_fuse.repeat_interleave(16, 0)
            attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]

            attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]

            # vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
            # an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 64).mean(dim=1)
            vn_hidden = vn_hidden + self.image_encoder.blocks[ii].gate * a2v_res_n
            an_hidden = an_hidden + self.audio_model.blocks[ii].gate * v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)

            vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.up_proj(vn_hidden)
            an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.up_proj(an_hidden)
            # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
            # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
            # self.drop_path = nn.Dropout(0.1)

            # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
            # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)

            v = v + self.image_encoder.blocks[ii].gate_fusion * vn_hidden  # self.drop_path(0.5 * )
            a = a + self.audio_model.blocks[ii].gate_fusion * an_hidden  # self.drop_path(0.5 *)


        print("ii:",ii,"   gate_fusion",self.image_encoder.blocks[ii].gate_fusion2)

        # print("ii:",ii,"   tanh(gate_fusion)",nn.functional.tanh(self.image_encoder.blocks[ii].gate_fusion))

        return v,a
# ABCabc123456...


    def fusion_adapter_simple(self,ii, v, a):



        B, L, C = v.shape
        # v = rearrange(v, '(b t) n c -> (b n) t c', t=16, n=L, c=C)
        # res_temporal = self.image_encoder.blocks[ii].attn(self.image_encoder.blocks[ii].norm1(v))
        # res_temporal = self.image_encoder.blocks[ii].T_Adapter(res_temporal, add_residual=False)
        # v = v + self.image_encoder.blocks[ii].drop_path1(res_temporal)
        # v = rearrange(v, '(b n) t c -> (b t) n c', t=16, n=L, c=C)

        # a = rearrange(a, '(b t) n c -> (b n) t c', t=1, n=L, c=C)
        # res_temporal_a = self.audio_model.blocks[ii].attn(self.audio_model.blocks[ii].norm1(a))
        # res_temporal_a = self.audio_model.blocks[ii].T_Adapter(res_temporal_a)
        # a = a + self.audio_model.blocks[ii].drop_path1(res_temporal_a)
        # a = rearrange(a, '(b n) t c -> (b t) n c', t=1, n=L, c=C)

        # self.drop_path = nn.Dropout(0.1)

        # 简化 adapter

        shortcut_v = v
        # x = self.norm1(v)

        shortcut_a = a
        # a = self.norm1(a)

        # attn_v = self.image_encoder.blocks[ii].attn(self.image_encoder.blocks[ii].norm1(v))
        # attn_a = self.audio_model.blocks[ii].attn(self.audio_model.blocks[ii].norm1(a))
        norm_v = self.image_encoder.blocks[ii].norm1(v)
        norm_a = self.audio_model.blocks[ii].norm1(a)

        attn_v = self.image_encoder.blocks[ii].attn(norm_v)
        attn_a = self.audio_model.blocks[ii].attn(norm_a)

        attn_v = self.image_encoder.blocks[ii].drop_path1(attn_v)
        attn_a = self.audio_model.blocks[ii].drop_path1(attn_a)

        if self.use_act== True :
            vs_hidden = self.image_encoder.blocks[ii].SAdapter2.non_linear_func(self.image_encoder.blocks[ii].SAdapter2.down_proj(norm_v))  # [n, bt, d]
            as_hidden = self.audio_model.blocks[ii].SAdapter2.non_linear_func(self.audio_model.blocks[ii].SAdapter2.down_proj(norm_a))
        else:
            vs_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter2.down_proj(norm_v))  # [n, bt, d]
            as_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter2.down_proj(norm_a))

        # # 加入layernorm尝试
        # vs_hidden = self.norm_layer(vs_hidden).to(vs_hidden.device)
        # as_hidden = self.norm_layer(as_hidden).to(as_hidden.device)





        # attn_v = attn_v + self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden)
        # attn_a = attn_a + self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden)

        vs_hidden = self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden)
        as_hidden = self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden)

        # attn_v = attn_v + self.image_encoder.blocks[ii].gate * as_hidden.repeat_interleave(16,0)
        # attn_a = attn_a + self.audio_model.blocks[ii].gate * vs_hidden.view(a.shape[0], 16, -1, 768).mean(1)
        attn_v = attn_v +  vs_hidden
        attn_a = attn_a +  as_hidden

        # attn_v = attn_v + self.image_encoder.blocks[ii].drop_path1(self.image_encoder.blocks[ii].SAdapter2.up_proj(vs_hidden))
        # attn_a = attn_a + self.audio_model.blocks[ii].drop_path1(self.audio_model.blocks[ii].SAdapter2.up_proj(as_hidden))


        # v = v.view(B_v, H * W, C)
        # x = shortcut + self.drop_path(x)
        # v = shortcut_v + self.image_encoder.blocks[ii].drop_path1(attn_v)
        v = shortcut_v + attn_v

        # a = a.view(B_a, H * W, C)
        # x = shortcut + self.drop_path(x)
        # a = shortcut_a + self.audio_model.blocks[ii].drop_path1(attn_a)
        a = shortcut_a + attn_a


        # FFN
        norm_vf = self.image_encoder.blocks[ii].norm2(v)
        norm_af = self.audio_model.blocks[ii].norm2(a)

        vn = self.image_encoder.blocks[ii].mlp(norm_vf)
        an = self.audio_model.blocks[ii].mlp(norm_af)

        # vn = self.image_encoder.blocks[ii].drop_path2(vn)
        # an = self.audio_model.blocks[ii].drop_path2(an)

        if self.use_act == True:
            vn_hidden = self.image_encoder.blocks[ii].adaptmlp.non_linear_func(self.image_encoder.blocks[ii].adaptmlp.down_proj(norm_vf))
            an_hidden = self.audio_model.blocks[ii].adaptmlp.non_linear_func(self.audio_model.blocks[ii].adaptmlp.down_proj(norm_af))
        else :
            vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].adaptmlp.down_proj(norm_vf))
            an_hidden = self.norm_layer(self.audio_model.blocks[ii].adaptmlp.down_proj(norm_af))

        # # 加入layernorm尝试
        # vn_hidden = self.norm_layer_late(vn_hidden).to(vn_hidden.device)
        # an_hidden = self.norm_layer_late(an_hidden).to(an_hidden.device)


        # vn_hidden = vn_hidden +  a2v_res_n.mean(1).unsqueeze(1)
        # an_hidden = an_hidden + v2a_res_n.view(a.shape[0], 16, -1, 128).mean(dim=1)

        vn_hidden = self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)
        # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
        # self.drop_path = nn.Dropout(0.1)

        vn = self.image_encoder.blocks[ii].drop_path2(vn)
        an = self.audio_model.blocks[ii].drop_path2(an)
        # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
        # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)

        # v = v + vn + self.image_encoder.blocks[ii].gate_fusion * an_hidden.repeat_interleave(16,0) # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].gate_fusion  * vn_hidden.view(a.shape[0], 16, -1, 768).mean(1)  # self.drop_path(0.5 *)
        v = v + vn + self.image_encoder.blocks[ii].drop_path2 (vn_hidden)   # self.drop_path(0.5 * )
        a = a + an + self.audio_model.blocks[ii].drop_path2(an_hidden)
              # self.drop_path(0.5 *)


        #feed 之前     feed后  上采样后
        # feed 后       上采样后


        # 在融合
        # FFN
        norm_vn = self.image_encoder.blocks[ii].norm2(v)
        norm_an = self.audio_model.blocks[ii].norm2(a)


        # vn = self.image_encoder.blocks[ii].drop_path2(vn)
        # an = self.audio_model.blocks[ii].drop_path2(an)

        if self.use_act == True:
            vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.non_linear_func(
                self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
            an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.non_linear_func(
                self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))
        else:
            vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter_fuse.down_proj(norm_vn))
            an_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter_fuse.down_proj(norm_an))

        # # 加入layernorm尝试
        # vn_hidden = self.norm_layer_late(vn_hidden).to(vn_hidden.device)
        # an_hidden = self.norm_layer_late(an_hidden).to(an_hidden.device)


        # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
        # self.drop_path = nn.Dropout(0.1)


        # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
        # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)
        vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse.up_proj(vn_hidden)
        an_hidden = self.audio_model.blocks[ii].SAdapter_fuse.up_proj(an_hidden)

        v = v  + self.image_encoder.blocks[ii].gate_fusion * vn_hidden # self.drop_path(0.5 * )
        a = a  + self.audio_model.blocks[ii].gate_fusion  * an_hidden  # self.drop_path(0.5 *)

        #按照图示 加的是 add之前的 所以是vn an
        # v = vn + self.image_encoder.blocks[ii].gate_fusion * vn_hidden  # self.drop_path(0.5 * )
        # a = an + self.audio_model.blocks[ii].gate_fusion * an_hidden  # self.drop_path(0.5 *)


        # v = v + vn + self.image_encoder.blocks[ii].drop_path2(vn_hidden)  # self.drop_path(0.5 * )
        # a = a + an + self.audio_model.blocks[ii].drop_path2(an_hidden)

        #
        # v = v + norm_vf
        # a = a + norm_af
        # v = self.image_encoder.blocks[ii].norm2(v)
        # a = self.audio_model.blocks[ii].norm2(a)

        if ii < 12 :
            #  尝试stack
            # FFN
            norm_vn = self.image_encoder.blocks[ii].norm2(v)
            norm_an = self.audio_model.blocks[ii].norm2(a)

            # vn = self.image_encoder.blocks[ii].drop_path2(vn)
            # an = self.audio_model.blocks[ii].drop_path2(an)

            if self.use_act == True:
                vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse2.non_linear_func(
                    self.image_encoder.blocks[ii].SAdapter_fuse2.down_proj(norm_vn))
                an_hidden = self.audio_model.blocks[ii].SAdapter_fuse2.non_linear_func(
                    self.audio_model.blocks[ii].SAdapter_fuse2.down_proj(norm_an))
            else:
                vn_hidden = self.norm_layer(self.image_encoder.blocks[ii].SAdapter_fuse2.down_proj(norm_vn))
                an_hidden = self.norm_layer(self.audio_model.blocks[ii].SAdapter_fuse2.down_proj(norm_an))


            vn_hidden = self.image_encoder.blocks[ii].SAdapter_fuse2.up_proj(vn_hidden)
            an_hidden = self.audio_model.blocks[ii].SAdapter_fuse2.up_proj(an_hidden)
            # v = v + vn + self.image_encoder.blocks[ii].adaptmlp.up_proj(vn_hidden)  # self.drop_path(0.5 * )
            # a = a + an + self.audio_model.blocks[ii].adaptmlp.up_proj(an_hidden)  # self.drop_path(0.5 *)
            # self.drop_path = nn.Dropout(0.1)

            print(vn_hidden.grad_fn)
            print(an_hidden.grad_fn)

            # v = v + vn +   self.image_encoder.blocks[ii].drop_path2(vn_hidden)   # self.drop_path(0.5 * )
            # a = a + an +    self.audio_model.blocks[ii].drop_path2(an_hidden) # self.drop_path(0.5 *)

            v = v  + self.image_encoder.blocks[ii].gate_fusion2 * an_hidden.repeat_interleave(16,0) # self.drop_path(0.5 * )
            a = a  + self.audio_model.blocks[ii].gate_fusion2  * vn_hidden.view(a.shape[0], 16, -1, 768).mean(1)  # self.drop_path(0.5 *)

            # v = v + self.image_encoder.blocks[ii].gate_fusion2 * vn_hidden  # self.drop_path(0.5 * )
            # a = a + self.audio_model.blocks[ii].gate_fusion2 * an_hidden  # self.drop_path(0.5 *)


        print("ii:",ii," gate_fusion1",self.image_encoder.blocks[ii].gate_fusion)
        print("gate_fusion1 grad:", self.image_encoder.blocks[ii].gate_fusion.grad)

        print("ii:",ii," gate_fusion2",self.image_encoder.blocks[ii].gate_fusion2)
        print("gate_fusion2 grad:", self.image_encoder.blocks[ii].gate_fusion2.grad)

        # print("ii:",ii,"   tanh(gate_fusion)",nn.functional.tanh(self.image_encoder.blocks[ii].gate_fusion))

        return v,a

    def modalfusion (self, image, audio):
        shortcut = audio
        audio = audio.repeat_interleave(16, 0)
        conv_mod1 = self.conv1d(image.transpose(1, 2))  # (B, 768, 197)
        conv_mod2 = self.conv1d(audio.transpose(1, 2))  # (B, 768, 197)

        # Apply sigmoid
        sigmoid_mod1 = torch.sigmoid(conv_mod1)
        sigmoid_mod2 = torch.sigmoid(conv_mod2)

        # Fusion by element-wise multiplication
        fused_mod1 = sigmoid_mod1 * image.transpose(1, 2)  # (B, 768, 197)
        fused_mod2 = sigmoid_mod2 * audio.transpose(1, 2)  # (B, 768, 197)

        # Transpose back to (B, 197, 768)
        fused_mod1 = fused_mod1.transpose(1, 2)
        fused_mod2 = fused_mod2.transpose(1, 2)

        fused_mod2 = fused_mod2.view(8, 16, 197, -1).mean(1)
        return image + fused_mod1, shortcut + fused_mod2

    def forward(self, image, audio):

        n, t, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        assert t == 16
        B = image.shape[0]

        # for ii in range(len(self.audio_model.blocks)):
        #     audio = self.audio_model.forward_block_pre(ii, audio) # torch.Size([B, 263, 768])
        #     image  = self.image_encoder.forward_block_pre(ii, image, B) #  torch.Size([B, 203, 768])  203 = 1(cls_token) + 196 + 6(prompts)
        #     # ABCabc123456...
        #     # 在此处添加 SimAM 模块
        #     # image = simam_module(image)
        #
        #     # image_lowdim_temp = self.image_encoder.temporal_pre[ii](image) # temporal_pre 线性层 降维
        #     # image_lowdim_norm = self.image_encoder.temporal_pre_norm[ii](image_lowdim_temp) # temporal_pre_norm layernorme
        #     #
        #     # audio_lowdim = self.image_encoder.audio_proj_pre[ii](audio) #  audio_proj_pre liner 768->128  + layernorm
        #     #
        #     # aaa = audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t, 0)
        #     #
        #     # bbb = image_lowdim_norm.view(B // t, t, self.n_image + 6 + 1, 128).mean(1).mean(1).unsqueeze(1)
        #       视觉 降维后 norm  +  音频降维的
        #     #
        #     # image_lowdim_norm2 = image_lowdim_norm + audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t,0)
        #     # audio_lowdim2 = audio_lowdim + image_lowdim_norm.view(B//t, t, self.n_image + 6 + 1, 128).mean(1).mean(1).unsqueeze(1)
        #
        #     image_lowdim_norm2 = torch.zeros(1).cuda()
        #     audio_lowdim2 =torch.zeros(1).cuda()
        #     # # 最后一轮 取 cls token
        #     image ,origin_image = self.image_encoder.forward_block_post(ii, image, image_lowdim_norm2, B)
        #     audio , origin_audio= self.audio_model.forward_block_post(ii, audio, audio_lowdim2)

        device = 'cuda:0'
        self.usePMF = True
        self.n_fusion_layers= 2
        if self.usePMF :

            for ii in range(len(self.audio_model.blocks) ):
            # for ii in range(len(self.audio_model.blocks) - self.n_fusion_layers):

                audio = self.audio_model.forward_block_pre(ii, audio)  # torch.Size([B, 263, 768])
                image = self.image_encoder.forward_block_pre(ii, image,
                                                             B)  # torch.Size([B, 203, 768])  203 = 1(cls_token) + 196 + 6(prompts)
                # image ,audio = self.fusion_adapter(ii, image, audio)
                image, audio = self.fusion_adapter_simple(ii, image, audio)

                # image, audio = self.modalfusion( image, audio)


                # ABCabc123456...
                # 在此处添加 SimAM 模块
                # image = simam_module(image)

                # image_lowdim_temp = self.image_encoder.temporal_pre[ii](image) # temporal_pre 线性层 降维
                # image_lowdim_norm = self.image_encoder.temporal_pre_norm[ii](image_lowdim_temp) # temporal_pre_norm layernorme
                #
                # audio_lowdim = self.image_encoder.audio_proj_pre[ii](audio) #  audio_proj_pre liner 768->128  + layernorm
                #
                # aaa = audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t, 0)
                #
                # bbb = image_lowdim_norm.view(B // t, t, self.n_image  + 1, 128).mean(1).mean(1).unsqueeze(1)
                #
                #
                # image_lowdim_norm2 = image_lowdim_norm + audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t,0)
                # audio_lowdim2 = audio_lowdim + image_lowdim_norm.view(B//t, t, self.n_image  + 1, 128).mean(1).mean(1).unsqueeze(1)

                image_lowdim_norm2 = torch.zeros(1).cuda()
                audio_lowdim2 = torch.zeros(1).cuda()
                # # 最后一轮 取 cls token
                image, origin_image = self.image_encoder.forward_block_post(ii, image, image_lowdim_norm2, B)
                audio, origin_audio = self.audio_model.forward_block_post(ii, audio, audio_lowdim2)

            txt_tokens = audio
            img_tokens = image

        #     ## multimodal fusion layers
        #     for fusion_layer_id in range(self.n_fusion_layers):
        #         ### get prompts
        #         batch_v2t_qp = self.v2t_qp[fusion_layer_id].expand(n * t , -1, -1).to(device)
        #         batch_t2v_qp = self.t2v_qp[fusion_layer_id].expand(n, -1, -1).to(device)
        #
        #         batch_v2t_qcp = self.v2t_qcp[fusion_layer_id].expand(n * t, -1, -1).to(device)
        #         batch_t2v_qcp = self.t2v_qcp[fusion_layer_id].expand(n, -1, -1).to(device)
        #
        #         if self.args.n_fcp > 0:
        #             batch_vision_fcp = self.vision_fcp[fusion_layer_id].expand(n * t, -1, -1).to(device)
        #             batch_text_fcp = self.text_fcp[fusion_layer_id].expand(n, -1, -1).to(device)
        #
        #         ### Query Stage
        #         # prepare text attn_mask
        #         ## slice attn_mask for corresponding text prompts
        #         # layer_t2v_qcp_attn_mask = batch_extra_attn_mask[:, :, :, :self.args.n_qcp]
        #         # layer_t2v_qp_attn_mask = batch_extra_attn_mask[:, :, :, :self.args.n_qp]
        #         # layer_text_fcp_attn_mask = batch_extra_attn_mask[:, :, :, :self.args.n_fcp]
        #         # layer_v2t_qp_attn_mask = batch_extra_attn_mask[:, :, :, :self.args.n_qp]
        #
        #         ## reform text attn_mask
        #         # query_txt_attn_mask = torch.cat([txt_attn_mask, layer_t2v_qcp_attn_mask, layer_t2v_qp_attn_mask], dim=3)
        #         # fusion_txt_attn_mask = torch.cat([txt_attn_mask, layer_text_fcp_attn_mask, layer_v2t_qp_attn_mask],
        #         #                                  dim=3)
        #
        #         # for t2v: get text fusion intermediate hidden-state for ViT
        #         query_txt_tokens = torch.cat([txt_tokens, batch_t2v_qcp, batch_t2v_qp], dim=1)
        #         t2v_fusion_intermediate = self.audio_model.blocks[ii + fusion_layer_id + 1](
        #             query_txt_tokens)
        #         t2v_fusion_intermediate = t2v_fusion_intermediate[:, -self.args.n_qp:, :]  # 要了cls的 qp
        #         t2v_fusion_intermediate = self.t2v_trans[fusion_layer_id](t2v_fusion_intermediate)  # 去做适配器
        #
        #         # for v2t: get vision fusion intermediate hidden-state for BERT
        #         query_img_tokens = torch.cat([img_tokens, batch_v2t_qcp, batch_v2t_qp], dim=1)
        #         v2t_fusion_intermediate = self.image_encoder.blocks[ii + fusion_layer_id + 1](
        #             query_img_tokens)
        #         v2t_fusion_intermediate = v2t_fusion_intermediate[:, -self.args.n_qp:, :]  # 要了cls的 qp
        #         v2t_fusion_intermediate = self.v2t_trans[fusion_layer_id](v2t_fusion_intermediate)  # 去做适配器
        #
        #         t2v_fusion_intermediate = t2v_fusion_intermediate.repeat_interleave(t, 0)
        #         v2t_fusion_intermediate = v2t_fusion_intermediate.view(n,t,-1,768).mean(dim=1)
        #         ### Fusion Stage
        #         img_tokens = torch.cat([img_tokens, batch_vision_fcp, t2v_fusion_intermediate], dim=1)
        #         txt_tokens = torch.cat([txt_tokens, batch_text_fcp, v2t_fusion_intermediate], dim=1)
        #
        #         img_tokens = self.image_encoder.blocks[ii + fusion_layer_id + 1](img_tokens)
        #         txt_tokens = \
        #         self.audio_model.blocks[ii + fusion_layer_id + 1](txt_tokens)
        #
        #         txt_tokens = txt_tokens[:, :-self.args.n_qp - self.args.n_fcp, :]
        #         img_tokens = img_tokens[:, :-self.args.n_qp - self.args.n_fcp, :]
        #
        # origin_audio = txt_tokens
        # origin_image = img_tokens



        # torch.Size([128, 203, 768]) origin_image
        # torch.Size([8, 263, 768]) origin_audio
        # torch.Size([16, 768]) image
        # torch.Size([1, 768]) audio
        # shortcut_v = origin_image
        # # origin_audio = origin_audio.repeat_interleave(t, 0)
        # origin_image = origin_image.reshape(n, t, -1, 768)  # Separate the 128 into 8 (batch) and 16 (repeats)
        #
        #
        # outputsimage = []
        # outputsaudio = []
        # for i in range(n):  # 每个batch 单独做ca  16帧视频 对应同一段音频
        #     imagetoken = origin_image[i]
        #     audiotokens = origin_audio[i].unsqueeze(0).repeat(t,1,1)
        #     # audiotokens = F.interpolate(origin_audio[i].unsqueeze(0), size=t, mode='linear')
        #     for ca in  self.ca_list:
        #         out_image, out_audio = ca(imagetoken, audiotokens)
        #     outputsimage.append(out_image)
        #     outputsaudio.append(out_audio)
        #
        #
        # image = torch.stack(outputsimage, dim=0).reshape(n*t, -1, 768)[:,0]
        # audio = torch.stack(outputsaudio, dim=0).mean(dim=1)[:,0]






        # self.gate_v

        # for ca in self.ca_list:
        # # #     origin_image = self.sa1(origin_image)
        # # #     origin_audio = self.sa2(origin_audio)
        #     origin_image, origin_audio = ca(origin_image, origin_audio)
        # src1 = origin_audio
        # src2 = origin_image

        # # src1 = self.norm_layer(src1).to(src1.device)
        # image = src1[:,0]
        #
        # # src2 = self.norm_layer(src2).to(src1.device)
        # src2 = src2.view(n, t, -1, 768)  # Separate the 128 into 8 (batch) and 16 (repeats)
        # src2 = src2.mean(dim=1)
        # audio = src2[:,0]

        # audio = audio.unsqueeze(0)

        # feat_fus = image + audio.unsqueeze(1)
        # feat_fus = feat_fus.permute(0, 2, 1).contiguous().to(device)
        # feat_fus = nn.AdaptiveAvgPool1d(1)(feat_fus).squeeze(2).to(device)
        # self.liner = nn.Linear(feat_fus.shape[1], 7).to(device)
        # feat_fus = self.liner(feat_fus).to(device)
        # return feat_fus



        # torch.Size([16, 768]) 取 cls tokens
        image = image.contiguous().view(n, t, -1) # torch.Size([B, 16, 768])
        image = self.vision_proj(image+audio.unsqueeze(1)) # torch.Size([B, 16, 512])

        video_features = self.temporal_net(image) # torch.Size([B, 512])  为 cls token

        output = self.our_classifier(video_features)

        return output

