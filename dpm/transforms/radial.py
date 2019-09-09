from .transform import Transform
from torch.nn import Parameter, init
from torch.nn.functional import softplus
import torch.nn as nn
import torch
import math
import dpm.utils as utils


class Radial(Transform):

    def __init__(self, n_dims=1):
        super().__init__()
        self.z_0 = Parameter(torch.zeros(n_dims))
        self.n_dims = n_dims
        self._alpha = Parameter(utils.softplus_inverse(torch.rand(1).float()))
        self.beta = Parameter(2+torch.randn(1).float())

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta_hat(self):
        return -self.alpha + softplus(self.beta)

    def h(self, z):
        return 1 / (self.alpha + self.r(z))

    def h_prime(self, z):
        return -1 / (self.alpha + self.r(z)) ** 2

    def r(self, z):
        return (z - self.z_0).abs()

    def forward(self, z):
        return z + self.beta_hat * self.h(z) * (z - self.z_0)

    def inverse(self, y):
        raise NotImplementedError("Direction not implemented for Radial")

    def log_abs_det_jacobian(self, z, y):
        bhp1 = 1 + self.beta_hat * self.h(z)
        term_1 = (self.n_dims - 1) * bhp1.log()
        term_2 = (bhp1 + self.beta_hat * self.h_prime(z) * self.r(z)).log()
        return term_1 + term_2
