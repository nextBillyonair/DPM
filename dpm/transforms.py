from abc import abstractmethod, ABC
import math
import torch
from torch.nn import Module, Parameter
from torch.nn.functional import softplus
import numpy as np


class Transform(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward not implemented")

    @abstractmethod
    def inverse(self, y):
        raise NotImplementedError("Inverse not implemented")

    @abstractmethod
    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError("Log Abs Det Jacobian not implemented")

    def softplus_inverse(self, value, threshold=20):
        inv = torch.log((value.exp() - 1.0))
        inv[value > threshold] = value[value > threshold]
        return inv

    def get_parameters(self):
        raise NotImplementedError('Get Parameters not implemented')


class InverseTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform.inverse(x)

    def inverse(self, y):
        return self.transform(y)

    def log_abs_det_jacobian(self, x, y):
        return -self.transform.log_abs_det_jacobian(y, x)

    def get_parameters(self):
        return {'type':'inverse'}


class Exp(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp()

    def inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x

    def get_parameters(self):
        return {'type':'exp'}


class Log(InverseTransform):

    def __init__(self):
        super().__init__(Exp())

    def get_parameters(self):
        return {'type':'log'}


# FIX
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


class Reciprocal(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1. / x

    def inverse(self, y):
        return 1. / y

    def log_abs_det_jacobian(self, x, y):
        return -2. * x.abs().log()

    def get_parameters(self):
        return {'type':'reciprocal'}


# Only works for +
class Square(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.pow(2)

    def inverse(self, y):
        return y.sqrt()

    def log_abs_det_jacobian(self, x, y):
        return np.log(2.) + x.log()

    def get_parameters(self):
        return {'type':'square'}


class Sigmoid(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def inverse(self, y):
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x) - softplus(x)
        # return -torch.log((y.reciprocal() + (1 - y).reciprocal()))

    def get_parameters(self):
        return {'type':'sigmoid'}


class Logit(InverseTransform):

    def __init__(self):
        super().__init__(Sigmoid())

    def get_parameters(self):
        return {'type':'logit'}


class Affine(Transform):

    def __init__(self, loc=0.0, scale=1.0, learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(1, -1)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(1, -1)
        self.loc = loc
        self.scale = scale
        self.n_dims = len(loc)
        if learnable:
            self.loc = Parameter(self.loc)
            self.scale = Parameter(self.scale)

    def forward(self, x):
        return self.loc + self.scale * x

    def inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        return torch.log(torch.abs(self.scale.expand(x.size())))

    def get_parameters(self):
        return {'type':'affine', 'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Expm1(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.expm1()

    def inverse(self, y):
        return y.log1p()

    def log_abs_det_jacobian(self, x, y):
        # log1p(y) = log1p(e^x - 1) = log((e^x - 1) + 1) = x
        return x

    def get_parameters(self):
        return {'type':'expm1'}


class Gumbel(Transform):

    def __init__(self, loc=0.0, scale=1.0, learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(1, -1)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(1, -1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def forward(self, x):
        z = (x - self.loc) / self.scale
        return torch.exp(-torch.exp(-z))

    def inverse(self, y):
        return self.loc - self.scale * torch.log(-torch.log(y))

    def log_abs_det_jacobian(self, x, y):
        return -torch.log(self.scale / (-torch.log(y) * y))

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        return {'type':'gumbel', 'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class SinhArcsinh(Transform):

    def __init__(self, skewness=0.0, tailweight=1.0, learnable=True):
        super().__init__()
        if not isinstance(skewness, torch.Tensor):
            skewness = torch.tensor(skewness).view(1, -1)
        if not isinstance(tailweight, torch.Tensor):
            tailweight = torch.tensor(tailweight).view(1, -1)
        self.skewness = skewness
        self._tailweight = self.softplus_inverse(tailweight)
        if learnable:
            self.skewness = Parameter(self.skewness)
            self._tailweight = Parameter(self._tailweight)

    def asinh(self, x):
        return torch.log(x + (x.pow(2) + 1).pow(0.5))

    def sqrtx2p1(self, x):
        return x.abs() * (1 + x.pow(-2)).sqrt()

    def forward(self, x):
        return torch.sinh((self.asinh(x) + self.skewness) * self.tailweight)

    def inverse(self, y):
        return torch.sinh(self.asinh(y) / self.tailweight - self.skewness)

    def log_abs_det_jacobian(self, x, y):
        return (torch.log(
                torch.cosh((self.asinh(x) + self.skewness) * self.tailweight)
                / self.sqrtx2p1(x + 1e-10)) + torch.log(self.tailweight))

    @property
    def tailweight(self):
        return softplus(self._tailweight)

    def get_parameters(self):
        return {'type':'sinharcsinh', 'skewness':self.skewness.detach().numpy(),
                'tailweight':self.tailweight.detach().numpy()}


class Softplus(Transform):

    def __init__(self, hinge_softness=1.0, learnable=True):
        super().__init__()
        if hinge_softness == 0.0: raise ValueError("Hinge Softness cannot be 0")
        if not isinstance(hinge_softness, torch.Tensor):
            hinge_softness = torch.tensor(hinge_softness).view(1, -1)
        self.hinge_softness = hinge_softness
        if learnable:
            self.hinge_softness = Parameter(self.hinge_softness)

    def forward(self, x):
        return self.hinge_softness * softplus(x / self.hinge_softness)

    def inverse(self, y):
        return self.hinge_softness * self.softplus_inverse(y / self.hinge_softness)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x / self.hinge_softness)
        # return torch.log(-torch.expm1(-y) + 1e-10)

    def get_parameters(self):
        return {'type':'softplus',
                'hinge_softness':self.hinge_softness.detach().numpy()}


class Softsign(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1.0 + x.abs())

    def inverse(self, y):
        return y / (1.0 - y.abs())

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * torch.log1p(-torch.abs(y))

    def get_parameters(self):
        return {'type':'softsign'}


class Tanh(Transform):

    def __init__(self):
        super().__init__()

    def atanh(self, x):
        return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

    def forward(self, x):
        return x.tanh()

    def inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2.0) - x - softplus(-2.0 * x))

    def get_parameters(self):
        return {'type':'tanh'}


# take forward, if not negate inverse
# EOF
