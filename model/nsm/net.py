import logging
import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata

# from .network import Expert, Encoder
from model.nsm.network import Expert, Encoder, SingleExpert

# Check GPU available
print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


class Model(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, segmentation,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For expert network information
                 expert_components, expert_dims, expert_activations, expert_dropout,
                 # optim param
                 lr,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.segmentation = segmentation

        # build encoder network
        self.encoder_nums = encoder_nums
        self.encoders = []
        for i in range(encoder_nums):
            encoder = Encoder(encoder_dims[i], encoder_activations[i], encoder_dropout)
            if torch.cuda.is_available():
                encoder = encoder.cuda()
            encoder = nn.DataParallel(encoder)
            self.encoders.append(encoder)

        # build gating network
        gating = Expert(expert_components[0], expert_dims[0], expert_activations[0], expert_dropout)
        if torch.cuda.is_available():
            gating = gating.cuda()
        gating = nn.DataParallel(gating)
        self.gating = gating

        # Original expert network
        # expert = Expert(expert_components[1], expert_dims[1], expert_activations[1], expert_dropout)
        # if torch.cuda.is_available():
        #     expert.cuda()
        # self.expert = expert

        # build expert network
        self.expert_nums = expert_components[-1]
        self.experts = []
        for i in range(self.expert_nums):
            expert = SingleExpert(expert_dims[-1], expert_activations[-1], expert_dropout)
            if torch.cuda.is_available():
                expert = expert.cuda()
            expert = nn.DataParallel(expert)
            self.experts.append(expert)

        # weight blend init
        self.weight_blend_init = torch.Tensor([1])
        if torch.cuda.is_available():
            self.weight_blend_init = self.weight_blend_init.cuda()

        # build optimizer
        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        params_list.append({'params': gating.parameters()})
        for e in self.experts:
            params_list.append({'params': e.parameters()})
        # params_list.append({'params': expert.parameters()})
        self.lr = lr
        self.optimizer = optim.AdamW(params_list,
                                     lr=self.lr)

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

    def load(self, load_path, e):
        print('Loading parm...')
        for i in range(self.encoder_nums):
            self.encoders[i].load_state_dict(torch.load(os.path.join(load_path, str(e)+'encoder%0i.pth' % i)))
        for i in range(self.expert_nums):
            self.experts[i].load_state_dict(torch.load(os.path.join(load_path, str(e)+'expert%0i.pth' % i)))
        self.gating.load_state_dict(torch.load(os.path.join(load_path, str(e)+'gating.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(load_path, str(e)+'optimizer.ptm')))
        print('Loading param complete')

    def to_eval(self):
        for i in range(self.encoder_nums):
            self.encoders[i].eval()
        for i in range(self.expert_nums):
            self.experts[i].eval()
        self.gating.eval()
        print('to eval...')

    def forward(self, x):
        status_outputs = []
        for i, encoder in enumerate(self.encoders):
            status_output = encoder(x[:, self.segmentation[i]:self.segmentation[i + 1]])
            status_outputs.append(status_output)
        status = torch.cat(tuple(status_outputs), 1)

        # Gating Network
        weight_blend_first = self.weight_blend_init.unsqueeze(0).expand(self.batch_size, 1)
        weight_blend = self.gating(weight_blend_first, x[:, self.segmentation[-2]:self.segmentation[-1]])

        # Expert Network
        outputs = torch.zeros(self.batch_size, 618).cuda()
        for index, net in enumerate(self.experts):
            expert_out = net(status)
            expert_out = expert_out * weight_blend[:, index].unsqueeze(-1)
            outputs = outputs + expert_out

        # Original Expert Network
        # outputs = self.expert(weight_blend, status)

        # Prediction

        return outputs

