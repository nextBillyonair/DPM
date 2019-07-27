import torch
from torch.nn import Parameter
from .transform import Transform


class Power(Transform):

    def __init__(self, power=1.0, learnable=True):
        super().__init__()
        if not isinstance(power, torch.Tensor):
            power = torch.tensor(power).view(1, -1)
        self.power = power
        if learnable:
            self.power = Parameter(self.power)

    def forward(self, x):
        if self.power == 0.:
            return x.exp()
        return (1. + x * self.power) ** (1. / self.power)

    def inverse(self, y):
        if self.power == 0.:
            return y.log()
        return (y**self.power - 1.) / self.power

    def log_abs_det_jacobian(self, x, y):
        if self.power == 0.:
            return x
        return (1. / self.power - 1.) * (x * self.power).log1p()

    def get_parameters(self):
        return {'type':'power', 'power':self.power.item()}
