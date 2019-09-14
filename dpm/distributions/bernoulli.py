import torch
from .distribution import Distribution
from .poisson import Poisson
from dpm.utils import eps, log
from torch.nn import Parameter
from torch._six import inf


class Bernoulli(Distribution):

    def __init__(self, probs=[0.5], learnable=True):
        super().__init__()
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs).view(-1)
        self.n_dims = len(probs)
        self.logits = log(probs.float())
        if learnable:
            self.logits = Parameter(self.logits)


    def log_prob(self, value):
        q = 1.-self.probs
        return (value * (self.probs + eps).log() + (1. - value) * (q + eps).log()).sum(-1)

    def sample(self, batch_size):
        return torch.bernoulli(self.probs.unsqueeze(0).expand((batch_size, *self.probs.shape)))

    def entropy(self):
        q = 1. - self.probs
        return -q * (q + eps).log() - self.probs * (self.probs + eps).log()

    def kl(self, other):
        if isinstance(other, Bernoulli):
            t1 = self.probs * (self.probs / other.probs).log()
            t1[other.probs == 0] = inf
            t1[self.probs == 0] = 0
            t2 = (1 - self.probs) * ((1 - self.probs) / (1 - other.probs)).log()
            t2[other.probs == 1] = inf
            t2[self.probs == 1] = 0
            return (t1 + t2).sum()
        if isinstance(other, Poisson):
            return (-self.entropy() - (self.probs * other.rate.log() - other.rate)).sum()
        return None

    @property
    def expectation(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    @property
    def skewness(self):
        return (1 - 2 * self.probs) / (self.probs * (1 - self.probs)).sqrt()

    @property
    def kurtosis(self):
        q = (1 - self.probs)
        return (1 - 6. * self.probs * q) / (self.probs * q)

    @property
    def probs(self):
        return self.logits.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'probs': self.probs.detach().item()}
        return {'probs': self.probs.detach().numpy()}
