import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import softplus
from .distribution import Distribution


class Uniform(Distribution):

    def __init__(self, low=0., high=1., learnable=True):
        super().__init__()
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low).view(-1)
        self.n_dims = len(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high).view(-1)
        self.alpha = low.float()
        self.beta = high.float()
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def log_prob(self, value):
        lb = value.ge(self.low).float()
        ub = value.lt(self.high).float()
        return (lb.mul(ub).log() - (self.high - self.low).log()).sum(-1)

    def sample(self, batch_size):
        u = torch.rand((batch_size, self.n_dims))
        return self.low + (self.high - self.low) * u

    def entropy(self, batch_size=None):
        return (self.high - self.low).log()

    def cdf(self, value):
        return ((value - self.low) / (self.high - self.low)).clamp(min=0, max=1)

    def icdf(self, value):
        return value * (self.high - self.low) + self.low

    @property
    def expectation(self):
        return (self.high + self.low) / 2

    @property
    def variance(self):
        return (self.high - self.low).pow(2) / 12

    @property
    def median(self):
        return self.expectation

    @property
    def low(self):
        return torch.min(self.alpha, self.beta)

    @property
    def high(self):
        return torch.max(self.alpha, self.beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'low':self.low.item(), 'high':self.high.item()}
        return {'low':self.low.detach().numpy(),
                'high':self.high.detach().numpy()}
