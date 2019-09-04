from .transform import Transform
from torch.nn import Parameter, init
from torch.nn.functional import softplus
import torch.nn as nn
import torch
import dpm.newton as newton
import dpm.utils as utils


class Radial(Transform):

    def __init__(self):
        self.z_0 = torch.randn(2)
        self._alpha = Parameter(utils.softplus_inverse(torch.rand().float()))
        self.beta = Parameter(torch.rand().float())

    @property
    def alpha(self):
        return softplus(self._alpha)

    def h(self, r):
        return (self.alpha + r).pow(-1)

    def r(self, z):
        return (x - self.z_0).pow(2).sum(-1).sqrt()

    def forward(self, x):
        return x + self.beta * self.h(self.r(x)) * (z - self.z_0)

    def inverse(self, y):
        raise NotImplementedError()

    def log_abs_det_jacobian(self, x, y):
        pass
