# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch

from timm.models.vision_transformer import VisionTransformer
# from models.vision_transformer import  Block ,VisionTransformer
from models.adapter import Adapter
import torch.nn as nn



from typing import Type


class VisionTransformer2(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_seq=196, n_progr=3, n_frames=16,**kwargs):
        super(VisionTransformer2, self).__init__(**kwargs)



        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm





    def forward_block_pre(self, ii, x, B):
        
        if ii == 0:
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # x = self.blocks[ii](x)
        return x

    def forward_block_post(self, ii, x, B):


        if ii == (len(self.blocks) - 1):
            if self.global_pool:
                x = x[:, 1:, :]
                x = x.mean(dim=1) 
                outcome = self.fc_norm(x)
                return outcome
            else:
                x = self.norm(x)
                outcome = x[:, 0]
                return outcome , x
        return x ,None #outcome


    def forward(self, x, ret_feature=False):
        x = self.forward_features(x)
        feature = x
        if getattr(self, 'head_dist', None) is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return
        if ret_feature:
            return x, feature
        else:
            return x


# setup model archs
VIT_KWARGS_BASE = dict(mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

VIT_KWARGS_PRESETS = {
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),
}

def create_vit_model(preset=None, creator=None, **kwargs):
    preset = 'base' if preset is None else preset.lower()
    all_kwargs = dict()
    all_kwargs.update(VIT_KWARGS_BASE)
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    all_kwargs.update(kwargs)
    if creator is None:
        creator = VisionTransformer2
    return creator(**all_kwargs)

#vit_tiny_patch16 = partial(create_vit_model, preset='tiny')
#vit_small_patch16 = partial(create_vit_model, preset='small')
vit_base_patch16 = partial(create_vit_model, preset='base')
#vit_large_patch16 = partial(create_vit_model, preset='large')
#vit_huge_patch14 = partial(create_vit_model, preset='huge')
#vit_giant_patch14 = partial(create_vit_model, preset='giant')
#vit_gigantic_patch14 = partial(create_vit_model, preset='gigantic')
