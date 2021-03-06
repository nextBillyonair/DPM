import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from .uniform import Uniform
from .transform_distribution import TransformDistribution
from dpm.transforms import Logit, Affine
import dpm.utils as utils

class Logistic(Distribution):

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

    def create_dist(self):
        zero = torch.zeros_like(self.loc)
        one = torch.ones_like(self.loc)
        model = TransformDistribution(Uniform(zero, one, learnable=False),
                                      [Logit(),
                                       Affine(self.loc, self.scale, learnable=False)])
        return model

    def log_prob(self, value):
        model = self.create_dist()
        return model.log_prob(value)

    def sample(self, batch_size):
        model = self.create_dist()
        return model.sample(batch_size)

    def cdf(self, value):
        model = self.create_dist()
        return model.cdf(value)

    def icdf(self, value):
        model = self.create_dist()
        return model.icdf(value)

    @property
    def scale(self):
        return softplus(self._scale)

    def entropy(self):
        return self.scale.log() + 2.

    @property
    def expectation(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self.scale.pow(2) * (math.pi**2) / 3

    @property
    def median(self):
        return self.loc

    @property
    def skewness(self):
        return torch.tensor(0.).float()

    @property
    def kurtosis(self):
        return torch.tensor(6./5.).float()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
