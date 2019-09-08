import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
import numpy as np
from .distribution import Distribution
from .uniform import Uniform
import dpm.utils as utils

class AsymmetricLaplace(Distribution):

    def __init__(self, loc=0., scale=1., asymmetry=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if not isinstance(asymmetry, torch.Tensor):
            asymmetry = torch.tensor(asymmetry).view(-1)
        self.loc = loc.float()
        self._scale = utils.softplus_inverse(scale.float())
        self._asymmetry = utils.softplus_inverse(asymmetry.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)
            self._asymmetry = Parameter(self._asymmetry)

    def log_prob(self, value):
        s = (value - self.loc).sign()
        exponent = -(value - self.loc).abs() * self.scale * self.asymmetry.pow(s)
        coeff = self.scale.log() - (self.asymmetry + (1 / self.asymmetry)).log()
        return (coeff + exponent).sum(-1)

    def sample(self, batch_size):
        U = Uniform(low=-self.asymmetry, high=(1./self.asymmetry), learnable=False).sample(batch_size)
        s = U.sign()
        log_term = (1. - U * s * self.asymmetry.pow(s)).log()
        return self.loc - (1. / (self.scale * s * self.asymmetry.pow(s))) * log_term

    def cdf(self, value):
        s = (value - self.loc).sign()
        exponent = -(value - self.loc).abs() * self.scale * self.asymmetry.pow(s)
        exponent = exponent.exp()
        return (value > self.loc).float() - s * self.asymmetry.pow(1 - s) / (1 + self.asymmetry.pow(2)) * exponent

    # def icdf(self, value):
    #     return

    def entropy(self):
        return (utils.e * (1 + self.asymmetry.pow(2)) / (self.asymmetry * self.scale)).log().sum()

    @property
    def expectation(self):
        return self.loc + ((1 - self.asymmetry.pow(2)) / (self.scale * self.asymmetry))

    @property
    def variance(self):
        return (1 + self.asymmetry.pow(4)) / (self.scale.pow(2) * self.asymmetry.pow(2))

    @property
    def mode(self):
        return self.loc

    @property
    def median(self):
        return self.loc + (self.asymmetry / self.scale) * ((1 + self.asymmetry.pow(2)) / (2 * self.asymmetry.pow(2))).log()

    @property
    def skewness(self):
        return (2 * (1 - self.asymmetry.pow(6))) / (1 + self.asymmetry.pow(4)).pow(3./2.)

    @property
    def kurtosis(self):
        return (6 * (1 + self.asymmetry.pow(8))) / (1 + self.asymmetry.pow(4)).pow(2)

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def asymmetry(self):
        return softplus(self._asymmetry)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(),
                    'scale':self.scale.item(),
                    'asymmetry':self.asymmetry.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy(),
                'asymmetry':self.asymmetry.detach().numpy()}
