import torch
from torch import nn
from einops import rearrange, reduce


class Time2Vector(nn.Module):
  def __init__(self, days:int, n_features:int, **kwargs):
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
    
    # mean = reduce(x[:,:,:self.n_features], 'b t f -> b t 1', 'mean')
    # time_linear = torch.einsum('b t m, t m -> b t', mean, self.weights_linear)
    # time_linear = rearrange(time_linear, 'b t -> b t 1')
    # time_linear = time_linear + self.bias_linear
    # time_linear = reduce(time_linear, 'b t f -> b t 1', 'sum')

    # time_periodic = torch.einsum('b t m, t m -> b t', x, self.weights_periodic)
    # time_periodic = rearrange(time_periodic, 'b t -> b t 1')
    # time_periodic = time_periodic + self.bias_periodic
    # time_periodic = reduce(time_periodic, 'b t f -> b t 1', 'sum')

    # time_periodic = torch.sin(time_periodic) 

    mean = reduce(x[:,:,:self.n_features], 'b t f -> b t', 'mean')

    time_linear = mean * self.weights_linear + self.bias_linear
    time_linear = rearrange(time_linear, 'b t -> b t 1')

    time_periodic = mean * self.weights_periodic + self.bias_periodic
    time_periodic = rearrange(time_periodic, 'b t -> b t 1')
    time_periodic = torch.sin(time_periodic)

    time_features = torch.cat([time_linear, time_periodic], dim=-1)

    return time_features # shape = (batch, days, 2)
