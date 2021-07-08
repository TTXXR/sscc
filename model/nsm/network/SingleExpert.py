import numpy as np
import torch
import torch.nn as nn
import os


class SingleExpert(nn.Module):
    def __init__(self, expert_dims, expert_activations, expert_dropout):
        super().__init__()
        self.expert_activations = expert_activations
        self.expert_dropout = expert_dropout
        self.layer_nums = len(expert_dims) - 1

        self.layer1 = nn.Sequential(nn.Dropout(expert_dropout),
                                    nn.Linear(expert_dims[0], expert_dims[1]),
                                    nn.ELU())
        self.layer2 = nn.Sequential(nn.Dropout(expert_dropout),
                                    nn.Linear(expert_dims[1], expert_dims[2]),
                                    nn.ELU())
        self.layer3 = nn.Sequential(nn.Dropout(expert_dropout),
                                    nn.Linear(expert_dims[2], expert_dims[3]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def save_network(self, expert_index, in_epoch):
        """
        save expert weight and bias for unity playing

        :param expert_index: this expert's index of all expert
        :param save_path: the root of save path
        :return: None
        """
        for i in range(self.layer_nums):
            self.state_dict()['layer%0i.1.weight' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join("C:/Users/rr/Desktop/sscc-main/model/nsm/model/part", str(in_epoch)+'expert%0i_w%0i.bin' % (expert_index, i)))
            self.state_dict()['layer%0i.1.bias' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join("C:/Users/rr/Desktop/sscc-main/model/nsm/model/part", str(in_epoch)+'expert%0i_b%0i.bin' % (expert_index, i)))
