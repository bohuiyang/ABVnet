
import torch.nn as nn

from typing import Optional
from torch import Tensor


class BEF(nn.Module):
    def __init__(self, channel, reduction=8):
        super(BEF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputt):
        x = inputt.permute(0, 2, 1).contiguous()
        b, c, f = x.size()
        gap = self.avg_pool(x).view(b, c)
        y = self.fc(gap).view(b, c, 1)
        out = x * y.expand_as(x)

        return out.permute(0, 2, 1).contiguous()


class SA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SA, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # MLP, used for FFN
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp = nn.Dropout(dropout)
        # self.linear_out = nn.Linear(dim_feedforward, d_model)
        # self.drop2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        self.se = BEF(channel=d_model, reduction=8)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                val=None):

        src_self = self.self_attention(src, src, value=val if val is not None else src,
                                       attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.drop1(src_self)
        src = self.norm1(src)
        # tmp = self.linear_out(self.dropout_mlp(self.activation(self.linear_in(src))))  # FFN

        tmp = self.se(src)
        src = self.norm2(src + self.drop1(tmp))

        return src


class CA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CA, self).__init__()
        # self.up = nn.Linear(d_model,)
        self.crs_attention1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.crs_attention2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP, used for FF
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in_1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_1 = nn.Dropout(dropout)
        # self.linear_out_1 = nn.Linear(dim_feedforward, d_model)
        # self.drop_1 = nn.Dropout(dropout)
        # self.norm_1 = nn.LayerNorm(d_model)

        # self.linear_in_2 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_2 = nn.Dropout(dropout)
        # self.linear_out_2 = nn.Linear(dim_feedforward, d_model)
        # self.drop_2 = nn.Dropout(dropout)
        # self.norm_2 = nn.LayerNorm(d_model)

        self.se1 = BEF(channel=d_model, reduction=8)    # 替换mlp可行
        self.se2 = BEF(channel=d_model, reduction=8)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                ):

        # print('src1.shape',src1.shape)
        # print('src2.shape', src2.shape)
        # src1.shape
        # torch.Size([128, 203, 768])
        # src2.shape
        # torch.Size([16, 263, 768])
        src1_cross = self.crs_attention1(query=src1,
                                         key=src2,
                                         value=src2, attn_mask=src2_mask,
                                         key_padding_mask=src2_key_padding_mask)[0]

        src2_cross = self.crs_attention2(query=src2,
                                         key=src1,
                                         value=src1, attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.drop1(src1_cross)
        src1 = self.norm1(src1)
        # tmp = self.linear_out_1(self.dropout_mlp_1(self.activation(self.linear_in_1(src1))))  # FFN

        tmp = self.se1(src1)
        src1 = self.norm1(src1 + self.drop1(tmp))

        src2 = src2 + self.drop2(src2_cross)
        src2 = self.norm2(src2)
        # tmp = self.linear_out_2(self.dropout_mlp_2(self.activation(self.linear_in_2(src2))))  # FFN

        tmp = self.se2(src2)
        src2 = self.norm2(src2 + self.drop2(tmp))

        return src1, src2

class adapterfusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CA, self).__init__()
        # self.up = nn.Linear(d_model,)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)


        # MLP, used for FF
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in_1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_1 = nn.Dropout(dropout)
        # self.linear_out_1 = nn.Linear(dim_feedforward, d_model)
        # self.drop_1 = nn.Dropout(dropout)
        # self.norm_1 = nn.LayerNorm(d_model)

        # self.linear_in_2 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_2 = nn.Dropout(dropout)
        # self.linear_out_2 = nn.Linear(dim_feedforward, d_model)
        # self.drop_2 = nn.Dropout(dropout)
        # self.norm_2 = nn.LayerNorm(d_model)

        self.se1 = BEF(channel=d_model, reduction=8)    # 替换mlp可行
        self.se2 = BEF(channel=d_model, reduction=8)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                ):

        # print('src1.shape',src1.shape)
        # print('src2.shape', src2.shape)
        # src1.shape
        # torch.Size([128, 203, 768])
        # src2.shape
        # torch.Size([16, 263, 768])
        src1_cross = self.attn(query=src1,
                                         key=src2,
                                         value=src2, attn_mask=src2_mask,
                                         key_padding_mask=src2_key_padding_mask)[0]

        src2_cross = self.crs_attention2(query=src2,
                                         key=src1,
                                         value=src1, attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.drop1(src1_cross)
        src1 = self.norm1(src1)
        # tmp = self.linear_out_1(self.dropout_mlp_1(self.activation(self.linear_in_1(src1))))  # FFN

        tmp = self.se1(src1)
        src1 = self.norm1(src1 + self.drop1(tmp))

        src2 = src2 + self.drop2(src2_cross)
        src2 = self.norm2(src2)
        # tmp = self.linear_out_2(self.dropout_mlp_2(self.activation(self.linear_in_2(src2))))  # FFN

        tmp = self.se2(src2)
        src2 = self.norm2(src2 + self.drop2(tmp))

        return src1, src2


class FusionNet(nn.Module):
    def __init__(self, backbone_dim=2048, c_dim=768, num_c=7):
        super(FusionNet, self).__init__()

        self.c_dim = c_dim  # 降维后的通道数
        self.backbone_dim = backbone_dim
        self.droprate = 0.3  # transformer的droprate
        self.nheads = 8
        self.dim_feedforward = 2048  # transformer中MLP的隐层节点数
        self.layers = 4
        # self.pos_rgb = PositionEmbeddingSine(c_dim // 2)
        # self.pos_depth = PositionEmbeddingSine(c_dim // 2)

        self.reduce_channel1 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.reduce_channel2 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_dim)
        self.bn2 = nn.BatchNorm2d(c_dim)

        self.sa1 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.sa2 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.ca_list = nn.ModuleList([CA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
                                      for _ in range(self.layers)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.1)

        self.fc_out1 = nn.Linear(c_dim, num_c)
        self.fc_out2 = nn.Linear(c_dim, num_c)
        self.fc_out3 = nn.Linear(c_dim, num_c)

        std = 0.001
        # normal(self.fc_out1.weight, 0, std)
        # constant(self.fc_out1.bias, 0)
        # normal(self.fc_out2.weight, 0, std)
        # constant(self.fc_out2.bias, 0)
        # normal(self.fc_out3.weight, 0, std)
        # constant(self.fc_out3.bias, 0)

    def forward(self, img_feat1, img_feat2):
        # 对channel做attention
        # img_feat1 = self.reduce_channel1(img_feat1)  # con1x1 减少通道数
        # img_feat2 = self.reduce_channel2(img_feat2)
        # img_feat1 = self.bn1(img_feat1)
        # img_feat2 = self.bn2(img_feat2)

        # # (N, L, E),where L is the target sequence length, N is the batch size, E is the embedding dimension.
        # img_feat1 = img_feat1.flatten(2).permute(0, 2, 1).contiguous()  # b f c
        # img_feat2 = img_feat2.flatten(2).permute(0, 2, 1).contiguous()

        # feat1 = img_feat1 + self.pos_rgb(img_feat1)
        # feat2 = img_feat2 + self.pos_depth(img_feat2)

        feat1 = img_feat1
        feat2 = img_feat2

        for ca in self.ca_list:
            feat1 = self.sa1(feat1)
            feat2 = self.sa2(feat2)
            feat1, feat2 = ca(feat1, feat2)

        # feat_fus = feat1 + feat2
        #
        # feat_fus = feat_fus.permute(0, 2, 1).contiguous()
        # img_feat1 = img_feat1.permute(0, 2, 1).contiguous()  # b c f
        # img_feat2 = img_feat2.permute(0, 2, 1).contiguous()
        #
        # feat_fus = self.avgpool(feat_fus).squeeze(2)
        # img_feat1 = self.avgpool(img_feat1).squeeze(2)  # 比conv1D好
        # img_feat2 = self.avgpool(img_feat2).squeeze(2)
        #
        # img_feat1 = self.drop1(img_feat1)
        # img_feat2 = self.drop2(img_feat2)
        # # feat_fus = self.drop3(feat_fus)
        #
        # img_feat1 = self.fc_out1(img_feat1)
        # img_feat2 = self.fc_out2(img_feat2)
        # feat_fus = self.fc_out3(feat_fus)

        return img_feat1, img_feat2, feat_fus

