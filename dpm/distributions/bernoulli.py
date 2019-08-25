import torch
from .distribution import Distribution
from dpm.utils import eps


class Bernoulli(Distribution):

    def __init__(self, probs=[0.5], learnable=False):
        super().__init__()
        self.n_dims = len(probs)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.float()
        self.logits = (self.probs + eps).log()

    def log_prob(self, value):
        q = 1.-self.probs
        return value * (self.probs + eps).log() + (1. - value) * (q + eps).log()

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

    def get_parameters(self):
        if self.n_dims == 1:
            return {'probs': self.probs.item()}
        return {'probs': self.probs.numpy()}
