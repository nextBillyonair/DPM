import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
import math
from .distribution import Distribution
import dpm.utils as utils

class Cauchy(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc.float()
        self._scale = utils.softplus_inverse(scale.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        ret = (1 + ((value - self.loc) / self.scale).pow(2)).log()
        return (-math.log(math.pi) - self.scale.log() - ret).sum(-1)

    def sample(self, batch_size):
        eps = torch.empty((batch_size, self.n_dims)).cauchy_()
        return self.loc + eps * self.scale

    def cdf(self, value):
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc

    def entropy(self, batch_size=None):
        return (4 * math.pi * self.scale).log()

    @property
    def median(self):
        return self.loc

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
