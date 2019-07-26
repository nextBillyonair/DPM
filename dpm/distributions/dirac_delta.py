import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class DiracDelta(Distribution):

    def __init__(self, loc=0., learnable=False):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
            if len(loc.shape) == 0:
                loc = loc.view(-1)
        self.n_dims = loc.shape
        self.loc = loc

    def log_prob(self, value):
        raise NotImplementedError("Dirac Delta log_prob not implemented")

    def sample(self, batch_size):
        return self.loc.expand(batch_size, *self.n_dims)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item()}
        return {'loc':self.loc.detach().numpy()}
