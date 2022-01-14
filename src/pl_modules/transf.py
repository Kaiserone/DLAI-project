import torch
from torch import nn
from torch.nn import functional as F


class Transf(nn.Module):
    def __init__(self, nhead: int, layers: int, time:int, n_feature: int) -> None:
        """
        Simple model that uses convolutions
        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_feature: size of the hidden dimensions to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.n_feature = n_feature
        layer = nn.TransformerEncoderLayer(
            d_model=n_feature, 
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

        #self.transformer = nn.ModuleList([nn.TransformerEncoder(layer, num_layers=time) for _ in range(layers)])
        self.fc1 = nn.Linear(time * n_feature, n_feature)
        self.fc2 = nn.Linear(n_feature, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :returns: predictions with size [batch, output_size]
        """
        #print(f"{x.shape=}")
        x = self.transformer(x)
        #print(f"{x.shape=}")
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        #print(f"{x.shape=}")

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        # print(x.shape)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.relu(x)
        return x