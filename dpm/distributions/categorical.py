import torch
from torch import distributions as dists
from torch.nn import Parameter
from torch.nn.functional import log_softmax
from .distribution import Distribution
from dpm.utils import log
from torch._six import inf

# convert to non dist version
# Differentiable log_prob, not differentiable sample
class Categorical(Distribution):

    def __init__(self, probs=[0.5, 0.5], learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = log(probs.float())
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = dists.Categorical(probs=self.probs)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = dists.Categorical(probs=self.probs)
        if len(self.probs.shape) != 1:
            return model.sample((batch_size, 1)).squeeze(1)
        return model.sample((batch_size, 1))

    def entropy(self):
        model = dists.Categorical(probs=self.probs)
        return model.entropy()

    def kl(self, other):
        if isinstance(other, Categorical):
            t = self.probs * (self.logits - other.logits)
            t[(other.probs == 0).expand_as(t)] = inf
            t[(self.probs == 0).expand_as(t)] = 0
            return t.sum()
        return None

    @property
    def probs(self):
        return self.logits.softmax(dim=-1)

    def get_parameters(self):
        return {'probs': self.probs.detach().numpy()}
