from .transform import Transform
from torch.nn import Parameter, init
from torch.nn.functional import softplus, elu
import torch.nn as nn
import torch
import math

class Planar(Transform):

    def __init__(self, in_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.w = Parameter(torch.Tensor(in_shape, 1))
        self.u = Parameter(torch.zeros(in_shape, 1))
        self.bias = Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        init.kaiming_uniform_(self.u, a=math.sqrt(5))

    def h(self, x):
        return torch.tanh(x)

    def h_prime(self, x):
        return 1 - torch.tanh(x).pow(2)

    @property
    def u_hat(self):
        w_u = self.w.t().mm(self.u)
        u_hat = self.u + (elu(w_u) - w_u) * self.w / (self.w.pow(2).sum())
        return u_hat

    def forward(self, z):
        return z + torch.mm((self.h(torch.mm(z, self.w) + self.bias)), self.u_hat.t())

    def inverse(self, z):
        raise NotImplementedError("Direction not implemented for Planar")

    def log_abs_det_jacobian(self, x, y):
        psi = torch.mm(x, self.w) + self.bias
        det_jacobian = 1 + self.h_prime(psi).mul(self.w.t().mm(self.u_hat))
        return det_jacobian.abs().log().sum(-1)




# EOF
