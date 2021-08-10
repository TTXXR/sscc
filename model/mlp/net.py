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

        # build mlp_network
        encoder = Encoder(self.encoder_dim, self.mlp_ratio, self.layer_num, self.encoder_dropout)
        self.encoder = encoder
        decoder = Decoder(self.decoder_dim, self.decoder_dropout)
        self.decoder = decoder

        # build optimizer
        self.lr = lr
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr)

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

    def load(self, load_path, e):
        self.encoder.load_state_dict(torch.load(
            os.path.join(load_path, str(e)+'encoder.pth'), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(
            os.path.join(load_path, str(e)+'decoder.pth'), map_location=lambda storage, loc: storage))

        self.encoder_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, str(e)+'encoder_optimizer.pth'),
                       map_location=lambda storage, loc: storage))
        self.decoder_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, str(e)+'decoder_optimizer.pth'),
                       map_location=lambda storage, loc: storage))

    def to_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
