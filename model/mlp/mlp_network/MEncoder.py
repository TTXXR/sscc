import torch.nn as nn
from .attention.gelu import GELU
from .Encoder import Mlp


class MEncoder(nn.Module):
    def __init__(self, dim, mlp_ratio, layer_num, dropout, act=GELU, norm=nn.LayerNorm):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.norm = norm(dim)
        self.attn_num = layer_num

        # MLP
        # self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=dropout)
        self.mlp_blocks = nn.ModuleList(
            [Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=dropout)
             for _ in range(self.attn_num)])

    def forward(self, x):
        for mlp in self.mlp_blocks:
            x = x + mlp(self.norm(x))

        # pure MLP
        # x = x + self.mlp1(self.norm(x))
        return x