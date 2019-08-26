import torch
from torch import distributions as dists
from .distribution import Distribution
from dpm.utils import eps

# convert to non dist version

class Categorical(Distribution):

    def __init__(self, probs=[0.5, 0.5], learnable=False):
        super().__init__()
        self.n_dims = len(probs)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.float()
        self.logits = (self.probs + eps).log()
        self.model = dists.Categorical(probs=probs)

    def log_prob(self, value):
        return self.model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        return self.model.sample((batch_size, 1))

    def entropy(self):
        return self.model.entropy()

    def get_parameters(self):
        return {'probs': self.probs.numpy()}
