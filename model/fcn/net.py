import torch
from model.fcn.MLP import MLP
from torch.nn import init
from model.ExternalAttention.ExternalAttention import ExternalAttention


class Model(object):
    def __init__(self, input_size=926, hidden_size=256, output_size=618,
                 learning_rate=0.1, batch_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.batch_size = batch_size

        self.attn_layer = ExternalAttention(self.input_size)
        self.model = MLP(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

        for params in self.model.parameters():
            init.normal_(params, mean=0, std=0.01)

        if torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda:0"))
            print("Using cuda:0.")

    def forward(self, x):
        # x = self.attn_layer(x)
        return self.model(x)
