import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
import math
from .distribution import Distribution
from dpm.utils import softplus_inverse, eps, euler_mascheroni

class Gumbel(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc.float()
        self._scale = softplus_inverse(scale.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return (-self.scale.log() - (z + (-z).exp())).sum(-1)

    def sample(self, batch_size):
        U = torch.rand((batch_size, self.n_dims))
        return self.icdf(U)

    def cdf(self, value):
        return (-(-(value - self.loc) / self.scale).exp()).exp()

    def icdf(self, value):
        return self.loc - self.scale * (-(value + eps).log()).log()

    def entropy(self):
        return self.scale.log() + euler_mascheroni + 1.

    @property
    def expectation(self):
        return self.loc + self.scale * euler_mascheroni

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return ((math.pi**2) / 6.) * self.scale.pow(2)

    @property
    def median(self):
        return self.loc - self.scale * math.log(math.log(2))

    @property
    def skewness(self):
        return torch.tensor(1.14).float() # expand this out

    @property
    def kurtosis(self):
        return torch.tensor(12./5.).float()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
