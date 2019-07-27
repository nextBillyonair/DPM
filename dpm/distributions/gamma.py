import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class Gamma(Distribution):

    def __init__(self, alpha=1., beta=1., learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta).view(-1)
        self._alpha = utils.softplus_inverse(alpha)
        self._beta = utils.softplus_inverse(beta)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def log_prob(self, value):
        return dists.Gamma(self.alpha, self.beta).log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        return dists.Gamma(self.alpha, self.beta).rsample((batch_size,))

    def entropy(self, batch_size=None):
        return dists.Gamma(self.alpha, self.beta).entropy()

    @property
    def expectation(self):
        return self.alpha / self.beta

    @property
    def variance(self):
        return self.alpha / self.beta.pow(2)

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}
