import torch
from torch.nn import Parameter
import dpm.utils as utils
from torch.nn.functional import softplus
from .distribution import Distribution
from .uniform import Uniform
from .transform_distribution import TransformDistribution
from dpm.transforms import Kumaraswamy as kumaraswamy_tform

class Kumaraswamy(Distribution):

    def __init__(self, alpha=1., beta=1., learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(1, -1)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta).view(1, -1)
        self._alpha = utils.softplus_inverse(alpha.float())
        self._beta = utils.softplus_inverse(beta.float())
        self.n_dims = len(alpha)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def create_dist(self):
        zero = torch.zeros_like(self._alpha)
        one = torch.ones_like(self._alpha)
        model = TransformDistribution(Uniform(zero, one, learnable=False),
                                      [kumaraswamy_tform(self.alpha, self.beta, learnable=False)])
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
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}
