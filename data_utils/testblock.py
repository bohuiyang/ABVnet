import torch
import torch.nn as nn
from typing import Optional
from timm.models.vision_transformer import Mlp ,Attention,LayerScale,DropPath
from models.adapter import Adapter
from einops import rearrange
from models.T_adapter import T_Adapter


class SAdapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = x + xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x

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
        self.adaptmlp = Adapter(d_model = 768, dropout=0.1, bottleneck=128,
                                init_option='lora',
                                adapter_scalar=1,
                                adapter_layernorm_option='in'
                                )

        # self.SAdapter2 = SAdapter2(dim)
        #
        # self.T_Adapter = T_Adapter(D_features=dim)

        # self.SAdapter2 = Adapter(d_model = 768, dropout=0.5, bottleneck=16,
        self.SAdapter2 = Adapter(d_model = 768, dropout=0.1, bottleneck=128,
                                init_option='lora',
                                adapter_scalar=1,
                                adapter_layernorm_option='',
                                 act_layer = nn.GELU
                                )

        self.SAdapter_fuse = Adapter(d_model=768, dropout=0.1, bottleneck=128,
                                 init_option='lora',
                                 adapter_scalar=1,
                                 adapter_layernorm_option='',
                                 act_layer=nn.GELU
                                 )
        # stack
        self.SAdapter_fuse2 = Adapter(d_model=768, dropout=0.1, bottleneck=128,
                                     init_option='lora',
                                     adapter_scalar=1,
                                     adapter_layernorm_option='',
                                     act_layer=nn.GELU
                                     )

        self.T_Adapter = Adapter(d_model = 768, dropout=0.1, bottleneck=128,
                                init_option='lora',
                                adapter_scalar=1,
                                adapter_layernorm_option='',
                                 act_layer=nn.GELU
                                )

        self.gate = nn.Parameter(torch.zeros(1))
        # self.gate_fusion = nn.Parameter(torch.zeros(1))

        # self.gate_fusion2 = nn.Parameter(torch.zeros(1))

        self.gate_fusion = nn.Parameter(torch.randn(1) * 0.001)

        self.gate_fusion2 = nn.Parameter(torch.randn(1) * 0.001)

        self.gate_hidden1 = nn.Parameter(torch.zeros(1))
        self.gate_hidden2 = nn.Parameter(torch.zeros(1))



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