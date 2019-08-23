import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class HalfCauchy(Distribution):

    def __init__(self, scale=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.n_dims = len(scale)
        self._scale = utils.softplus_inverse(scale.float())
        if learnable:
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        return dists.HalfCauchy(self.scale).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return dists.HalfCauchy(self.scale).rsample((batch_size,))

    def entropy(self, batch_size=None):
        return dists.HalfCauchy(self.scale).entropy()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'scale':self.scale.item()}
        return {'scale':self.scale.detach().numpy()}
