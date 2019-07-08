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


class Exp(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp()

    def inverse(self, y):
        return torch.log(y)

    def log_abs_det_jacobian(self, x, y):
        return x

    def get_parameters(self):
        return {'type':'exp'}


class Power(Transform):

    def __init__(self, exponent=1.0, learnable=True):
        super().__init__()
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent).view(1, -1)
        self.exponent = exponent
        if learnable:
            self.exponent = Parameter(self.exponent)

    def forward(self, x):
        return x.pow(self.exponent)

    def inverse(self, y):
        return y.pow(1 / self.exponent)

    def log_abs_det_jacobian(self, x, y):
        # return torch.log((self.exponent * y / x).abs())
        if self.exponent == 1:
            return torch.zeros_like(x)
        return torch.log(self.exponent.abs()) + (self.exponent - 1) * torch.log(x.abs())

    def get_parameters(self):
        return {'type':'power', 'exponent':self.exponent.item()}


class Reciprocal(Power):

    def __init__(self):
        super().__init__(-1.0, learnable=False)


class Sigmoid(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def inverse(self, y):
        return torch.log(y) - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -torch.log((y.reciprocal() + (1 - y).reciprocal()))

    def get_parameters(self):
        return {'type':'sigmoid'}


class Logit(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x) - (-x).log1p()

    def inverse(self, y):
        return torch.sigmoid(y)

    def log_abs_det_jacobian(self, x, y):
        return torch.log((x.reciprocal() + (1 - x).reciprocal()))

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
        return torch.log(-torch.expm1(-y) + 1e-10)

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
