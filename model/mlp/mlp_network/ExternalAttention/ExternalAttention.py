import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):
    def __init__(self, input_size, drop):
        super(ExternalAttention, self).__init__()
        self.input_size = input_size

        # reshaped
        self.extend_size = 4096
        self.hidden_size = 64
        self.n = 8

        self.extend = nn.Linear(self.input_size, self.extend_size)
        self.proj = nn.Linear(self.extend_size, self.input_size)
        self.Mk = nn.Linear(self.extend_size//self.n, self.hidden_size, bias=False)  # 1024 256
        self.Mv = nn.Linear(self.hidden_size, self.extend_size//self.n, bias=False)

        # origin
        # self.Mk = nn.Linear(input_size, self.hidden_size, bias=False)
        # self.Mv = nn.Linear(self.hidden_size, input_size, bias=False)

        self.softmax = nn.Softmax(dim=-2)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

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
        # reshaped
        x = self.extend(x)
        x = x.view(x.shape[0], self.n, self.extend_size//self.n)

        # origin
        attn = self.Mk(x)  # F:N*d-8*1024 M:S*d-256*1024
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # norm
        x = self.Mv(attn)  # F:N*d-8*256 M:S*d-1024*256

        # reshaped
        x = x.view(x.shape[0], self.extend_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
