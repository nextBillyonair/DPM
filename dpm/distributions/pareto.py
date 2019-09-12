import torch
from torch.nn import Parameter
from .transform_distribution import TransformDistribution
from .distribution import Distribution
from .exponential import Exponential
from dpm.transforms import Affine, Exp
from dpm.utils import softplus_inverse
from torch.nn.functional import softplus

class Pareto(Distribution):

    def __init__(self, scale=1., alpha=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)

        self._scale = softplus_inverse(scale.float())
        self._alpha = softplus_inverse(alpha.float())
        self.n_dims = len(scale)

        if learnable:
            self._scale = Parameter(self._scale)
            self._alpha = Parameter(self._alpha)

    def create_dist(self):
        model = TransformDistribution(Exponential(self.alpha, learnable=False),
                [Exp(), Affine(torch.zeros_like(self._scale), self.scale, learnable=False)])
        return model

    def log_prob(self, value):
        model = self.create_dist()
        return model.log_prob(value)

    def sample(self, batch_size):
        model = self.create_dist()
        return model.sample(batch_size)

    def cdf(self, value):
        model = self.create_dist()
        return model.cdf(value)

    def icdf(self, value):
        model = self.create_dist()
        return model.icdf(value)

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def expectation(self):
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1.)

    @property
    def median(self):
        return self.scale * torch.tensor(2.).pow(1 / self.alpha)

    @property
    def mode(self):
        return self.scale

    @property
    def variance(self):
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    def entropy(self):
        return ((self.scale / self.alpha).log() + (1 + self.alpha.reciprocal()))

    def get_parameters(self):
        return {'scale':self.scale.detach().numpy(),
                'alpha':self.alpha.detach().numpy()}
