import torch
from torch import distributions as dists
from torch.nn import Parameter
from .distribution import Distribution
from dpm.utils import eps

# convert to non dist version
# Differentiable log_prob, not differentiable sample
class Categorical(Distribution):

    def __init__(self, probs=[0.5, 0.5], learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = probs.float().log()
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = dists.Categorical(probs=self.probs)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = dists.Categorical(probs=self.probs)
        return model.sample((batch_size, 1))

    def entropy(self):
        model = dists.Categorical(probs=self.probs)
        return model.entropy()

    @property
    def probs(self):
        return self.logits.softmax(dim=-1)

    def get_parameters(self):
        return {'probs': self.probs.detach().numpy()}
