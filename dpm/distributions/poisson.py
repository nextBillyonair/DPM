import torch
from torch.nn import Parameter
from torch.nn.functional import softplus
from .distribution import Distribution
from dpm.utils import softplus_inverse


class Poisson(Distribution):

    def __init__(self, rate=1., learnable=True):
        super().__init__()
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).view(-1)
        self._rate = softplus_inverse(rate.float())
        self.n_dims = len(rate)
        if learnable:
            self._rate = Parameter(self._rate)

    def log_prob(self, value):
        return (self.rate.log() * value - self.rate - (value + 1).lgamma()).sum(-1)

    def sample(self, batch_size):
        return torch.poisson(self.rate.expand((batch_size, self.n_dims)))

    @property
    def expectation(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    @property
    def median(self):
        return torch.floor(self.rate + (1. / 3.) - 0.02 / self.rate)

    @property
    def mode(self):
        return self.rate.floor()

    @property
    def skewness(self):
        return self.rate.pow(-0.5)

    @property
    def kurtosis(self):
        return self.rate.pow(-1)

    @property
    def rate(self):
        return softplus(self._rate)

    def get_parameters(self):
        return {'rate':self.rate.detach().numpy()}
