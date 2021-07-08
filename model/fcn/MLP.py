import torch
from torch import nn
from torch.nn import init


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.fc = nn.Sequential(
        #     # FlattenLayer,
        #     nn.Linear(self.input_size, self.hidden_size),
        #     # nn.BatchNorm1d(self.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.output_size),
        # )

        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.input_size, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.input_size),
            # nn.BatchNorm1d(self.input_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.input_size, self.output_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
