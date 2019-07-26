import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution

# Uses dist + transforms
class TransformDistribution(Distribution):

    def __init__(self, distribution, transforms, learnable=False):
        super().__init__()
        self.n_dims = distribution.n_dims
        self.distribution = distribution
        self.transforms = ModuleList(transforms)

    def log_prob(self, value):
        prev_value = value
        log_det = 0.0
        for transform in self.transforms[::-1]:
            value = transform.inverse(prev_value)
            log_det += transform.log_abs_det_jacobian(value, prev_value)
            prev_value = value
        return -log_det.sum(1) + self.distribution.log_prob(value)

    def sample(self, batch_size):
        samples = self.distribution.sample(batch_size)
        for transform in self.transforms:
            samples = transform(samples)
        return samples

    def get_parameters(self):
        return {'distribution':self.distribution.get_parameters(),
                'transforms': [transform.get_parameters()
                               for transform in self.transforms]}
