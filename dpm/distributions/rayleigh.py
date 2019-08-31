import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
from .distribution import Distribution
import dpm.utils as utils
import math

class Rayleigh(Distribution):
    def __init__(self, scale=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.n_dims = len(scale)
        self._scale = utils.softplus_inverse(scale.float())
        if learnable:
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        log_exponent = -value.pow(2) / (2 * self.scale.pow(2))
        return (value.log() - 2 * self.scale.log() + log_exponent).sum(-1)

    def sample(self, batch_size):
        u = torch.rand((batch_size, self.n_dims))
        return self.scale * (-2 * u.log()).sqrt()

    def entropy(self, batch_size=None):
        return 1. + (self.scale / math.sqrt(2.)).log() + utils.euler_mascheroni / 2.

    def cdf(self, value):
        exponent = -value.pow(2) / (2 * self.scale.pow(2))
        return 1 - exponent.exp()

    def icdf(self, value):
        return self.scale * (-2 * (-value).log1p()).sqrt()

    @property
    def expectation(self):
        return self.scale * math.sqrt(math.pi / 2.)

    @property
    def variance(self):
        return ((4. - math.pi) / 2.) * self.scale.pow(2)

    @property
    def median(self):
        return self.scale * math.sqrt((2. * math.log(2.)))

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'scale':self.scale.item()}
        return {'scale':self.scale.detach().numpy()}
