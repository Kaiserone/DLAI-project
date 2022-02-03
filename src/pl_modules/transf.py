import torch
from torch import nn

class Transformer(nn.Module):
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
        norm = nn.BatchNorm1d(time)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, norm=norm)
       
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :returns: predictions with size [batch, output_size]
        """
        return self.transformer(x)