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

from .mlp_network import Encoder, Decoder


class Model(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, save_path, load_path,
                 # For Date information
                 train_source, test_source,
                 # For encoder mlp_network information
                 segmentation, encoder_dim, encoder_num, mlp_ratio, encoder_dropout,
                 # For decoder mlp_network information
                 decoder_dim, decoder_dropout,
                 # optim param
                 lr, layer_num
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.save_path = save_path
        self.load_path = load_path

        self.train_source = train_source
        self.test_source = test_source

        self.segmentation = segmentation
        self.encoder_dim = encoder_dim
        self.mlp_ratio = mlp_ratio
        self.layer_num = layer_num
        self.encoder_dropout = encoder_dropout
        self.decoder_dim = decoder_dim
        self.decoder_dropout = decoder_dropout

        # build mult attention for all features
        self.encoder_num = encoder_num
        self.encoders = []
        for i in range(self.encoder_num):
            encoder = Encoder(int(self.segmentation[i+1] - self.segmentation[i]), self.mlp_ratio, self.layer_num, self.encoder_dropout)
            self.encoders.append(encoder)

        decoder = Decoder(self.decoder_dim, self.decoder_dropout)
        self.decoder = decoder

        # build optimizer
        self.lr = lr
        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        self.encoder_optimizer = torch.optim.AdamW(params_list, lr=self.lr)
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr)

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

    def load(self, load_path, e):
        for i, encoder in enumerate(self.encoders):
            encoder.module.load_state_dict(torch.load(
                os.path.join(self.load_path, str(e) + 'encoder' + str(i) + '.pth'),
                map_location=lambda storage, loc: storage), False)
        self.decoder.module.load_state_dict(torch.load(
            os.path.join(load_path, str(e)+'decoder.pth'), map_location=lambda storage, loc: storage), False)

        self.encoder_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, str(e)+'encoder_optimizer.pth'),
                       map_location=lambda storage, loc: storage))
        self.decoder_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, str(e)+'decoder_optimizer.pth'),
                       map_location=lambda storage, loc: storage))

    def to_eval(self):
        for i, encoder in enumerate(self.encoders):
            encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        # mult attention for all features
        outputs = []
        for i, encoder in enumerate(self.encoders):
            output = encoder(x[:, self.segmentation[i]: self.segmentation[i+1]])
            outputs.append(output)
        x = torch.cat(tuple(outputs), dim=1)

        x = self.decoder(x)
        return x
