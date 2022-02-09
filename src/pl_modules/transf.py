import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, nhead: int, layers: int, days:int, n_features: int) -> None:
        """
        Simple model that uses transformers
        :param nhead: multi-head number
        :param layers: number of hidden layers of the transformer
        :param days: length of the sequence
        :param n_features: size of the input features
        """
        super().__init__()
        self.n_features = n_features
        layer = nn.TransformerEncoderLayer(
            d_model=n_features, 
            nhead=nhead,
            batch_first=True
        )
        norm = nn.BatchNorm1d(days)
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