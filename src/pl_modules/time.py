import torch
from torch import nn
from einops import rearrange, reduce


class Time2Vector(nn.Module):
  def __init__(self, days:int, n_features:int, **kwargs):
    """
      Simple model that uses transformers
      :param days: length of the sequence
      :param n_features: size of the input features
    """
    super(Time2Vector, self).__init__()
    self.days = days
    self.n_features = n_features
    seq_shape = (days,)

    '''Initialize weights and biases with shape (batch, days)'''
    self.weights_linear = nn.Parameter(torch.empty(size=seq_shape, dtype=torch.float32).uniform_(0, 1))
    
    self.bias_linear = nn.Parameter(torch.empty(size=seq_shape, dtype=torch.float32).uniform_(0, 1))
    
    self.weights_periodic = nn.Parameter(torch.empty(size=seq_shape, dtype=torch.float32).uniform_(0, 1))
    
    self.bias_periodic = nn.Parameter(torch.empty(size=seq_shape, dtype=torch.float32).uniform_(0, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''Calculate linear and periodic days features'''

    mean = reduce(x[:,:,:self.n_features], 'b t f -> b t', 'mean') #mean of the first n_features

    time_linear = mean * self.weights_linear + self.bias_linear #linear function
    time_linear = rearrange(time_linear, 'b t -> b t 1') #expand

    time_periodic = mean * self.weights_periodic + self.bias_periodic #linear component for periodic features
    time_periodic = rearrange(time_periodic, 'b t -> b t 1') #expand
    time_periodic = torch.sin(time_periodic) #apply periodic function

    time_features = torch.cat([time_linear, time_periodic], dim=-1)

    return time_features # shape = (batch, days, 2)
