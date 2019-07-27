import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class Exponential(Distribution):

    def __init__(self, rate=1., learnable=True):
        super().__init__()
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).view(-1)
        self.n_dims = len(rate)
        self._rate = utils.softplus_inverse(rate)
        if learnable:
            self._rate = Parameter(self._rate)

    def log_prob(self, value):
        return dists.Exponential(self.rate).log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        return dists.Exponential(self.rate).rsample((batch_size,))

    def entropy(self, batch_size=None):
        return 1 - self.rate.log()

    def cdf(self, value):
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -(1 - value).log() / self.rate

    @property
    def expectation(self):
        return self.rate.pow(-1)

    @property
    def variance(self):
        return self.rate.pow(-2)

    @property
    def median(self):
        return self.rate.pow(-1) * np.log(2.)

    @property
    def rate(self):
        return softplus(self._rate)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'rate':self.rate.item()}
        return {'rate':self.rate.detach().numpy()}
