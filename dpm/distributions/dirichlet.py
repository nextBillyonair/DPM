import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class Dirichlet(Distribution):

    def __init__(self, alpha=[0.5, 0.5], learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        self._alpha = utils.softplus_inverse(alpha.float())
        if learnable:
            self._alpha = Parameter(self._alpha)

    def log_prob(self, value):
        return dists.Dirichlet(self.alpha).log_prob(value)

    def sample(self, batch_size):
        return dists.Dirichlet(self.alpha).rsample((batch_size,))

    def entropy(self):
        return dists.Dirichlet(self.alpha).entropy()

    @property
    def alpha(self):
        return softplus(self._alpha)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item()}
        return {'alpha':self.alpha.detach().numpy()}
