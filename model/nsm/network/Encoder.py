import os
import torch
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self, encoder_dims, encoder_activations, encoder_dropout):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.encoder_dropout = encoder_dropout
        self.layer_nums = len(encoder_dims) - 1

        self.layer1 = nn.Sequential(nn.Dropout(encoder_dropout),
                                    nn.Linear(encoder_dims[0], encoder_dims[1]),
                                    nn.ELU())
        self.layer2 = nn.Sequential(nn.Dropout(encoder_dropout),
                                    nn.Linear(encoder_dims[1], encoder_dims[2]),
                                    nn.ELU())
        self.layer3 = nn.Sequential(nn.Dropout(encoder_dropout),
                                    nn.Linear(encoder_dims[2], encoder_dims[3]),
                                    nn.ELU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x

    def save_network(self, encoder_index, in_epoch):
        for i in range(self.layer_nums):
            self.state_dict()['layer%0i.1.weight' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join("C:/Users/rr/Desktop/sscc-main/model/nsm/model/part", str(in_epoch)+'encoder%0i_w%0i.bin' % (encoder_index, i)))
            self.state_dict()['layer%0i.1.bias' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join("C:/Users/rr/Desktop/sscc-main/model/nsm/model/part", str(in_epoch)+'encoder%0i_b%0i.bin' % (encoder_index, i)))
