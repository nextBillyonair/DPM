import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Arcsine(Distribution):

    def __init__(self, low=0., high=1., learnable=True):
        super().__init__()
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low).view(-1)
        self.n_dims = len(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high).view(-1)
        self.alpha = low
        self.beta = high
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def log_prob(self, value):
        return - (math.pi * ((value - self.low) * (self.high - value)).sqrt()).log().reshape(-1)

    def sample(self, batch_size):
        raise NotImplementedError("sample not implemented")

    @property
    def low(self):
        return torch.min(self.alpha, self.beta)

    @property
    def high(self):
        return torch.max(self.alpha, self.beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'low':self.low.item(), 'high':self.high.item()}
        return {'low':self.low.detach().numpy(),
                'high':self.high.detach().numpy()}