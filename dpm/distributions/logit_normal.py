import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
import numpy as np
from .distribution import Distribution
from .normal import Normal
from .transform_distribution import TransformDistribution
from dpm.transforms import Sigmoid
import dpm.utils as utils

class LogitNormal(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        self.model = TransformDistribution(Normal(loc, scale, learnable=learnable),
                                           [Sigmoid()])
        self.n_dims = self.model.n_dims

    def log_prob(self, value):
        return self.model.log_prob(value)

    def sample(self, batch_size):
        return self.model.sample(batch_size)

    def cdf(self, value):
        return self.model.cdf(value)

    def icdf(self, value):
        return self.model.icdf(value)

    @property
    def scale(self):
        return self.model.distribution.scale

    @property
    def loc(self):
        return self.model.distribution.loc

    @property
    def median(self):
        return torch.sigmoid(self.loc)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
