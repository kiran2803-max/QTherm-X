import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    """

    def __init__(self):

        super(PINN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        return self.network(x)