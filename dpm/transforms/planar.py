from .transform import Transform
from torch.nn import Parameter, init
from torch.nn.functional import softplus
import torch.nn as nn
import torch
import math

class Planar(Transform):

    def __init__(self, in_shape=1, activation='Tanh'):
        super().__init__()
        self.in_shape=in_shape
        self.w = Parameter(torch.Tensor(in_shape, 1))
        self.u = Parameter(torch.Tensor(in_shape, 1))
        self.bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        init.kaiming_uniform_(self.u, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def h(self, x):
        return torch.tanh(x)

    def h_prime(self, x):
        return 1 - torch.tanh(x).pow(2)

    @property
    def u_hat(self):
        w_u = self.w.t().mm(self.u)
        u_hat = self.u + (softplus(w_u) - 1. - w_u) * self.w / (self.w.pow(2).sum())
        return u_hat

    def forward(self, y):
        return y + torch.mm((self.h(torch.mm(y, self.w) + self.bias)), self.u_hat.t())

    def inverse(self, x):
        raise NotImplementedError('Planar Flow direction not implemented')

    def log_abs_det_jacobian(self, x, y):
        psi = torch.mm(y, self.w) + self.bias
        return torch.log(1 + self.h_prime(psi).mul(self.w.t().mm(self.u_hat)))




# EOF
