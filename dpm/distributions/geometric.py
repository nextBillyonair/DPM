import torch
from torch.nn import Parameter
from .distribution import Distribution
from dpm.utils import log


# second on on wiki
# Models num of failures before first success
class Geometric(Distribution):

    def __init__(self, probs=1., learnable=True):
        super().__init__()
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs).view(-1)
        self.n_dims = len(probs)
        self.probs = probs.float()
        if learnable:
            self.probs = Parameter(self.probs)

    def log_prob(self, value):
        return (value * (-self.probs).log1p() + log(self.probs)).sum(-1)

    def sample(self, batch_size):
        u = torch.rand((batch_size, self.n_dims))
        return (u.log() / (-self.probs).log1p()).floor()

    def cdf(self, value):
        return 1 - (1 - self.probs).pow(value + 1.)

    def icdf(self, value):
        return ((-value).log1p() / (-self.probs).log1p()) - 1.

    def entropy(self):
        q = (1. - self.probs)
        return - (q * utils.log(q) + self.probs * utils.log(self.probs)) / self.probs

    def kl(self, other):
        if isinstance(other, Geometric):
            return (-self.entropy() - (-other.probs).log1p() / self.probs - other.logits).sum()
        return None

    @property
    def expectation(self):
        return 1. / self.probs - 1.

    @property
    def variance(self):
        return (1. / self.probs - 1.) / self.probs

    @property
    def mode(self):
        return torch.tensor(0.).float()

    @property
    def skewness(self):
        return (2 - self.probs) / (1 - self.probs).sqrt()

    @property
    def kurtosis(self):
        return 6. + (self.probs.pow(2) / (1. - self.probs))

    @property
    def median(self):
        return (-1 / (-self.probs).log1p()).ceil() - 1.

    @property
    def logits(self):
        return log(self.probs)

    def get_parameters(self):
        return {'probs':self.probs.detach().numpy()}
