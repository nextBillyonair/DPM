from .transform import Transform
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch
import dpm.utils as utils


class Weibull(Transform):

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

    def forward(self, x):
        return  -(-((x / self.scale).pow(self.concentration))).expm1()

    def inverse(self, y):
        return self.scale * (-(-y).log1p()).pow(1 / self.concentration)

    def log_abs_det_jacobian(self, x, y):
        return (-(x / self.scale).pow(self.concentration) \
            + (self.concentration * utils.log(x)) + self.concentration.log() \
            - self.concentration * self.scale.log())

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def concentration(self):
        return softplus(self._concentration)

    def get_parameters(self):
        return {'type':'weibull',
                'scale':self.scale.detach().numpy(),
                'concentration':self.concentration.detach().numpy()}
