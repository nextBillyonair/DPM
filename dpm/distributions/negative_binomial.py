import torch
from torch.nn import Parameter
from torch.distributions import NegativeBinomial as NB
from .distribution import Distribution
from dpm.utils import log


class NegativeBinomial(Distribution):

    def __init__(self, total_count=10, probs=[0.5], learnable=True):
        super().__init__()
        if not isinstance(probs, torch.Tensor):
            total_count = torch.tensor(total_count)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs).view(-1)
        self.n_dims = len(probs)
        self.total_count = total_count
        self.logits = log(probs.float())
        if learnable:
            self.total_count = Parameter(self.total_count)
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        return NB(self.total_count, probs=self.probs).log_prob(value)

    def sample(self, batch_size):
        return NB(self.total_count, probs=self.probs).sample((batch_size, ))

    @property
    def expectation(self):
        return (self.probs * self.total_count) / (1 - self.probs)

    @property
    def mode(self):
        if self.total_count > 1:
            return (self.probs * (self.total_count - 1) / (1 - self.probs)).floor()
        return torch.tensor(0.).float()

    @property
    def variance(self):
        return self.probs * self.total_count / (1 - self.probs).pow(2)

    @property
    def skewness(self):
        return (1 + self.probs) / (self.probs * self.total_count).sqrt()

    @property
    def kurtosis(self):
        return 6. / self.total_count + (1 - self.probs).pow(2) / (self.probs * self.total_count)

    @property
    def probs(self):
        return self.logits.exp()

    def get_parameters(self):
        return {'probs':self.probs.detach().numpy()}
