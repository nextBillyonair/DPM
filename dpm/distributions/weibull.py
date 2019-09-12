import torch
from torch.nn import Parameter
import dpm.utils as utils
from torch.nn.functional import softplus
from .distribution import Distribution
from .uniform import Uniform
from .transform_distribution import TransformDistribution
from dpm.transforms import Weibull as weibull_tform
from dpm.transforms import InverseTransform

class Weibull(Distribution):

    def __init__(self, scale=1., concentration=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(1, -1)
        if not isinstance(concentration, torch.Tensor):
            concentration = torch.tensor(concentration).view(1, -1)
        self._scale = utils.softplus_inverse(scale.float())
        self._concentration = utils.softplus_inverse(concentration.float())
        self.n_dims = len(scale)
        if learnable:
            self._scale = Parameter(self._scale)
            self._concentration = Parameter(self._concentration)

    def create_dist(self):
        zero = torch.zeros_like(self._scale)
        one = torch.ones_like(self._scale)
        model = TransformDistribution(Uniform(zero, one, learnable=False),
                    [InverseTransform(weibull_tform(self.scale, self.concentration, learnable=False))])
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
    def concentration(self):
        return softplus(self._concentration)

    def get_parameters(self):
        return {'scale':self.scale.detach().numpy(),
                'concentration':self.concentration.detach().numpy()}
