import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import softplus
import numpy as np
from torch import distributions as dists
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
        self.loc = loc.float()
        self._scale = utils.softplus_inverse(scale.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        return (-(2. * self.scale).log() - ((value - self.loc).abs() / self.scale)).sum(-1)

    def sample(self, batch_size):
        return dists.Laplace(self.loc, self.scale).rsample((batch_size,))

    def cdf(self, value):
        return 0.5 - 0.5 * (value - self.loc).sign() * (-(value - self.loc).abs() / self.scale).expm1()

    def icdf(self, value):
        term = value - 0.5
        return self.loc - self.scale * term.sign() * (-2 * term.abs()).log1p()

    def entropy(self):
        return 1 + (2 * self.scale).log()

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
    def stddev(self):
        return (2 ** 0.5) * self.scale

    @property
    def mode(self):
        return self.loc

    @property
    def skewness(self):
        return torch.tensor(0.).float()

    @property
    def kurtosis(self):
        return torch.tensor(3.).float()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
