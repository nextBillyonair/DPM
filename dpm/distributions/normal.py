import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Normal(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if scale.shape == loc.shape:
            scale = torch.diag(scale)
        self.loc = loc
        self.cholesky_decomp = scale.view(self.n_dims, self.n_dims).cholesky()
        if learnable:
            self.loc = Parameter(self.loc)
            self.cholesky_decomp = Parameter(self.cholesky_decomp)

    def log_prob(self, value):
        return dists.MultivariateNormal(self.loc, self.scale).log_prob(value)

    def sample(self, batch_size):
        return dists.MultivariateNormal(self.loc, self.scale).rsample((batch_size,))

    def entropy(self, batch_size=None):
        return 0.5 * torch.det(2 * math.pi * math.e * self.scale).log()

    @property
    def expectation(self):
        return self.loc

    @property
    def variance(self):
        return self.scale

    @property
    def scale(self):
        return torch.mm(self.cholesky_decomp, self.cholesky_decomp.t())

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
