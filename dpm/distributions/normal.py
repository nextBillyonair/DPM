import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from dpm.utils import softplus_inverse


class Normal(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).float()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).float()

        if len(loc.shape) == 0:
            loc = loc.view(-1)
            scale = scale.view(-1)
            self.n_dims = 1
            self._scale = softplus_inverse(scale)
            self._diag_type = 'diag'

        if len(loc.shape) == 1:
            self.n_dims = len(loc)
            scale = scale.view(-1)
            if scale.numel() == 1:
                scale = scale.expand_as(loc)

            if scale.shape == loc.shape:
                self._scale = softplus_inverse(scale)
                self._diag_type = 'diag'
            else:
                self._scale = scale.view(self.n_dims, self.n_dims).cholesky()
                self._diag_type = 'cholesky'

            self.loc = loc

        if len(loc.shape) > 1:
            assert len(loc.shape) == len(scale.shape)
            self.loc = loc

            scale = scale.expand_as(loc)
            self._diag_type = 'diag'
            self._scale = softplus_inverse(scale)
            self.n_dims = loc.shape

        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        if self._diag_type == "cholesky":
            return dists.MultivariateNormal(self.loc, self.scale).log_prob(value)
        elif self._diag_type == 'diag':
            return dists.Normal(self.loc, self.std).log_prob(value).sum(dim=-1)
        else:
            raise NotImplementedError("_diag_type can only be cholesky or diag")

    def sample(self, batch_size):
        if self._diag_type == "cholesky":
            return dists.MultivariateNormal(self.loc, self.scale).rsample((batch_size,))
        elif self._diag_type == 'diag':
            return dists.Normal(self.loc, self.std).rsample((batch_size, ))
        else:
            raise NotImplementedError("_diag_type can only be cholesky or diag")

    def entropy(self, batch_size=None):
        if self._diag_type == "cholesky":
            return 0.5 * torch.det(2 * math.pi * math.e * self.scale).log()
        elif self._diag_type == 'diag':
            return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
        else:
            raise NotImplementedError("_diag_type can only be cholesky or diag")

    def cdf(self, value):
        if self._diag_type == 'diag':
            return dists.Normal(self.loc, self.std).cdf(value)
        else:
            raise NotImplementedError("CDF only implemented for _diag_type diag")

    def icdf(self, value):
        if self._diag_type == 'diag':
            return dists.Normal(self.loc, self.std).icdf(value)
        else:
            raise NotImplementedError("CDF only implemented for _diag_type diag")

    def kl(self, other):
        if isinstance(other, Normal):
            if other._diag_type == 'diag': # regular normal
                var_ratio = (self.scale / other.scale).pow(2)
                t1 = ((self.loc - other.loc) / other.scale).pow(2)
                return (0.5 * (var_ratio + t1 - 1. - var_ratio.log())).sum()
        return None

    @property
    def expectation(self):
        return self.loc

    @property
    def variance(self):
        return self.scale

    @property
    def std(self):
        return torch.diagonal(self.scale, dim1=-2, dim2=-1).sqrt()

    @property
    def scale(self):
        if self._diag_type == 'cholesky':
            return torch.mm(self._scale, self._scale.t())
        elif self._diag_type == 'diag':
            return torch.diag_embed(softplus(self._scale))
        else:
            raise NotImplementedError("_diag_type can only be cholesky or diag")

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
