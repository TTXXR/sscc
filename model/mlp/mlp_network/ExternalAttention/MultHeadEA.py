import torch
from torch import nn
from torch.nn import init


class MultHeadEA(nn.Module):
    def __init__(self, input_size, drop, num_heads, coef=4):
        super().__init__()
        # reshaped
        self.input_size = input_size
        self.num_heads = num_heads
        self.coef = coef
        self.stand_size = 4096
        self.k = 256//self.coef

        self.n = 8
        # ex_attn
        self.trans_dim = nn.Linear(self.input_size, self.stand_size*self.coef)
        self.num_heads = self.num_heads*self.coef

        self.Mk = nn.Linear(self.stand_size*self.coef//self.n//self.num_heads, self.k, bias=False)
        self.Mv = nn.Linear(self.k, self.stand_size * self.coef // self.n // self.num_heads, bias=False)

        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.stand_size * self.coef, self.input_size)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.trans_dim(x)
        b, n_f = x.shape  # n_f:self.n*self.num_heads*...
        x = x.contiguous().view(b, self.n, n_f//self.n)
        x = x.contiguous().view(b, self.n, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.Mk(x)
        attn = self.softmax(attn)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # norm
        attn = self.attn_drop(attn)

        x = self.Mv(attn).permute(0, 2, 1, 3).view(b, self.n, -1)
        x = x.view(b, -1)
        proj = self.proj(x)
        proj = self.proj_drop(proj)

        return proj
