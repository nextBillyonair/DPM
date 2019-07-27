import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from .cauchy import Cauchy
from .transform_distribution import TransformDistribution
from dpm.transforms import Exp
import dpm.utils as utils


class LogCauchy(Distribution):

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
        model = TransformDistribution(Cauchy(self.loc, self.scale, learnable=False),
                                      [Exp()])
        return model.log_prob(value)

    def sample(self, batch_size):
        model = TransformDistribution(Cauchy(self.loc, self.scale, learnable=False),
                                      [Exp()])
        return model.sample(batch_size)

    def cdf(self, value):
        std_term = (value.log() - self.loc) / self.scale
        return (1. / math.pi) * torch.atan(std_term) + 0.5

    def icdf(self, value):
        cauchy_icdf = torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc
        return cauchy_icdf.exp()

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def median(self):
        return self.loc.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
