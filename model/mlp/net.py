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

# Check GPU available
print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


class Model(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size,
                 # For encoder mlp_network information
                 encoder_dim, mlp_ratio, encoder_dropout,
                 # For decoder mlp_network information
                 decoder_dim, decoder_dropout,
                 # optim param
                 lr,
                 # attn layer num
                 layer_num,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size

        self.encoder_dim = encoder_dim
        self.mlp_ratio = mlp_ratio
        self.layer_num = layer_num
        self.encoder_dropout = encoder_dropout
        self.decoder_dim = decoder_dim
        self.decoder_dropout = decoder_dropout

        # build mlp_network
        encoder = Encoder(self.encoder_dim, self.mlp_ratio, self.layer_num, self.encoder_dropout)
        if torch.cuda.is_available():
            encoder = nn.DataParallel(encoder.cuda())
        self.encoder = encoder

        decoder = Decoder(self.decoder_dim, self.decoder_dropout)
        if torch.cuda.is_available():
            decoder = nn.DataParallel(decoder.cuda())
        self.decoder = decoder

        # build optimizer
        self.lr = lr
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr)

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

    def up_lr(self):
        pass

    def save(self, save_path, e):
        # Save Model
        torch.save(self.encoder.state_dict(), os.path.join(save_path, str(e)+"encoder.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, str(e)+"decoder.pth"))
        # Save optimizer
        torch.save(self.encoder_optimizer.state_dict(), os.path.join(save_path, str(e)+"encoder_optimizer.pth"))
        torch.save(self.decoder_optimizer.state_dict(),
                   os.path.join(save_path, str(e)+"decoder_optimizer.pth"))

    def load(self, load_path, e):
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, str(e)+'encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(load_path, str(e)+'decoder.pth')))

        self.encoder_optimizer.load_state_dict(torch.load(os.path.join(load_path, str(e)+'encoder_optimizer.pth')))
        self.decoder_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, str(e)+'decoder_optimizer.pth')))

    def to_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
