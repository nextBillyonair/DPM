import torch
from .distribution import Distribution
from dpm.utils import log, e, pi
from torch.nn import Parameter
from torch.distributions.binomial import Binomial as BN


class Binomial(Distribution):

    def __init__(self, total_count=10, probs=[0.5], learnable=True):
        super().__init__()
        if not isinstance(probs, torch.Tensor):
            total_count = torch.tensor(total_count)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs).view(-1)
        self.n_dims = len(probs)
        self.total_count = total_count.float()
        self.logits = log(probs.float())
        if learnable:
            self.total_count = Parameter(self.total_count)
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        return BN(self.total_count, probs=self.probs).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return BN(self.total_count, probs=self.probs).sample((batch_size, ))

    def entropy(self):
        return 0.5 * (2 * pi * e * self.total_count * self.probs * (1-self.probs)).log()

    @property
    def expectation(self):
        return self.total_count * self.probs

    @property
    def mode(self):
        return ((self.total_count  + 1) * self.probs).floor()

    @property
    def median(self):
        return (self.total_count * self.probs).floor()

    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    @property
    def skewness(self):
        return (1 - 2 * self.probs) / (self.total_count * self.probs * (1 - self.probs)).sqrt()

    @property
    def kurtosis(self):
        pq = self.probs * (1 - self.probs)
        return (1 - 6 * pq) / (self.total_count * pq)

    @property
    def probs(self):
        return self.logits.exp()

    def get_parameters(self):
        return {'total_count':self.total_count.detach().numpy(),
                'probs':self.probs.detach().numpy()}




#EOF
