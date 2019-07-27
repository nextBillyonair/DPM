import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class Laplace(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = utils.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        return dists.Laplace(self.loc, self.scale).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return dists.Laplace(self.loc, self.scale).rsample((batch_size,))

    def cdf(self, value):
        return dists.Laplace(self.loc, self.scale).cdf(value)

    def icdf(self, value):
        return dists.Laplace(self.loc, self.scale).icdf(value)

    def entropy(self, batch_size=None):
        return (2 * self.scale * math.e).log()

    @property
    def expectation(self):
        return self.loc

    @property
    def variance(self):
        return 2 * self.scale.pow(2)

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
