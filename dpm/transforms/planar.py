from .transform import Transform
from torch.nn import Parameter, init
import torch.nn as nn
import torch
import dpm.newton as newton

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

    def forward(self, y):
        return y + torch.mm((self.h(torch.mm(y, self.w) + self.bias)), self.u.t())

    def inverse(self, x):
        raise NotImplementedError('Planar Flow does not have forward')

    def log_abs_det_jacobian(self, x, y):
        psi = torch.mm(y, self.w) + self.bias
        return 1 + self.h_prime(psi).mul(self.w.t().mm(self.u))




# EOF
