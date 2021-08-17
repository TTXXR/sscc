import torch.nn as nn
from .attention.gelu import GELU
from .ExternalAttention import ExternalAttention
from .ExternalAttention import MultHeadEA


class Encoder(nn.Module):
    def __init__(self, dim, mlp_ratio, layer_num, dropout, act=GELU, norm=nn.LayerNorm):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.norm = norm(dim)
        self.attn_num = layer_num

        # MLP
        # self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=dropout)
        # mult head EA
        self.mh_blocks = nn.ModuleList(
            [MultHeadEA(dim, dropout, num_heads=8) for _ in range(self.attn_num)])
        # self.mult_head_attn = MultHeadEA(dim, dropout, num_heads=8)
        # external attention
        # self.attn_blocks = nn.ModuleList(
        #     [ExternalAttention(input_size=dim, drop=dropout) for _ in range(self.attn_num)])

    def forward(self, x):
        # external attention
        # for attn in self.attn_blocks:
        #     x = x + attn(x)

        # pure MLP
        # x = x + self.mlp1(self.norm(x))

        for block in self.mh_blocks:
            x = x + block(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=GELU, drop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
