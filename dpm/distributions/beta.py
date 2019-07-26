import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Beta(Distribution):

    def __init__(self, alpha=0.5, beta=0.5, learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta).view(-1)
        self._alpha = self.softplus_inverse(alpha)
        self._beta = self.softplus_inverse(beta)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def log_prob(self, value):
        return dists.Beta(self.alpha, self.beta).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return dists.Beta(self.alpha, self.beta).rsample((batch_size,))

    def entropy(self, batch_size=None):
        return dists.Beta(self.alpha, self.beta).entropy()

    @property
    def expectation(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total).pow(2) * (total + 1))

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha': self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}
