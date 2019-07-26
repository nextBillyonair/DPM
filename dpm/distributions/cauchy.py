import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Cauchy(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        return dists.Cauchy(self.loc, self.scale).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return dists.Cauchy(self.loc, self.scale).rsample((batch_size,))

    def cdf(self, value):
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc

    def entropy(self, batch_size=None):
        return (4 * math.pi * self.scale).log()

    @property
    def median(self):
        return self.loc

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
