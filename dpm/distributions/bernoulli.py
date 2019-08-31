import torch
from .distribution import Distribution
from dpm.utils import eps
from torch.nn import Parameter


class Bernoulli(Distribution):

    def __init__(self, probs=[0.5], learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.float()
        if learnable:
            self.probs = Parameter(self.probs)


    def log_prob(self, value):
        q = 1.-self.probs
        return (value * (self.probs + eps).log() + (1. - value) * (q + eps).log()).sum(-1)

    def sample(self, batch_size):
        return torch.bernoulli(self.probs.unsqueeze(0).expand((batch_size, *self.probs.shape)))

    def entropy(self):
        q = 1. - self.probs
        return -q * (q + eps).log() - self.probs * (self.probs + eps).log()

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
    def logits(self):
        return (self.probs + eps).log()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'probs': self.probs.detach().item()}
        return {'probs': self.probs.detach().numpy()}
