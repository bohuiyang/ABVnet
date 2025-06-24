# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn

from typing import Optional
from timm.models.vision_transformer import Mlp ,Attention,LayerScale,DropPath
from einops import rearrange


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 act_layer=nn.ReLU
                 ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = act_layer()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)


        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output



class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            use_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.config.d = None
        if use_adapter == True:
            self.G_Adapter = Adapter(d_model = 768, dropout=0.1, bottleneck=128,
                                    init_option='lora',
                                    adapter_scalar=1,
                                    adapter_layernorm_option='',
                                    act_layer = nn.GELU
                                    )
            #  adaptmlp   SAdapter2  SAdapter_fuse SAdapter_fuse2

            self.S_Adapter = Adapter(d_model = 768, dropout=0.1, bottleneck=128,
                                    init_option='lora',
                                    adapter_scalar=1,
                                    adapter_layernorm_option='',
                                     act_layer = nn.GELU
                                    )

        self.Adapter1 = Adapter(d_model=768, dropout=0.1, bottleneck=128,
                                 init_option='lora',
                                 adapter_scalar=1,
                                 adapter_layernorm_option='',
                                 act_layer=nn.GELU
                                 )

        self.Adapter2 = Adapter(d_model=768, dropout=0.1, bottleneck=128,
                                     init_option='lora',
                                     adapter_scalar=1,
                                     adapter_layernorm_option='',
                                     act_layer=nn.GELU
                                     )


        self.gate_fusion1 = nn.Parameter(torch.randn(1) * 0.1)
        self.gate_fusion2 = nn.Parameter(torch.randn(1) * 0.1)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('进入block')

        B, L, C = x.shape
        # if B == 16:  #torch.Size([16, 203, 768])
        #     x = rearrange(x, '(b t) n c -> (b n) t c', t=16, n=L, c=C)
        #     res_temporal = self.attn(self.norm1(x))
        #     res_temporal = self.T_Adapter(res_temporal, add_residual=False)
        #     x = x + self.drop_path1(res_temporal)
        #     x = rearrange(x, '(b n) t c -> (b t) n c', t=16, n=L, c=C)
        #
        #
        #     shortcut_v = x
        #     x = self.norm1(x)
        #
        #     # shortcut_a = audio
        #     # audio = self.norm1(audio)
        #
        #     shortcut = x
        # # print('进入block')
        #
        #     res_spatial = self.attn(self.norm1(x))
        #     res_spatial = self.SAdapter2(res_spatial, add_residual=False)
        #     # x = x + self.drop_path1(self.ls1(res_spatial))
        #
        #     x = x + res_spatial
        #
        #
        #
        #
        # else :
        #     x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        adapt_x = self.adaptmlp(x, add_residual=False)

        # residual = x
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        #残差结
        x = x + adapt_x
        # x = residual + x

        return x


class ResPostBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))


        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x