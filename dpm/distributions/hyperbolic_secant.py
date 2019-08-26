import torch
import math
from .distribution import Distribution
import dpm.utils as utils

# generalize for loc + scale?
class HyperbolicSecant(Distribution):

    def __init__(self, n_dims=1, learnable=False):
        super().__init__()
        self.n_dims = n_dims

    def log_prob(self, value):
        return -math.log(2.) + utils.sech((math.pi / 2.) * value).log()

    def sample(self, batch_size):
        U = torch.rand((batch_size, self.n_dims))
        return self.icdf(U)

    def cdf(self, value):
        return (2. / math.pi) * utils.arctan(((math.pi / 2.) * value).exp())

    def icdf(self, value):
        return (2. / math.pi) * (((math.pi / 2.) * value).tan()).log()

    def entropy(self, batch_size=None):
        return (4. / math.pi) * utils.catalan

    @property
    def expectation(self):
        return torch.tensor(0.)

    @property
    def variance(self):
        return torch.tensor(1.)

    @property
    def median(self):
        return torch.tensor(0.)
