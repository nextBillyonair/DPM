from .transform import Transform
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch
import dpm.utils as utils


class SinhArcsinh(Transform):

    def __init__(self, skewness=0.0, tailweight=1.0, learnable=True):
        super().__init__()
        if not isinstance(skewness, torch.Tensor):
            skewness = torch.tensor(skewness).view(1, -1)
        if not isinstance(tailweight, torch.Tensor):
            tailweight = torch.tensor(tailweight).view(1, -1)
        self.skewness = skewness
        self._tailweight = utils.softplus_inverse(tailweight)
        if learnable:
            self.skewness = Parameter(self.skewness)
            self._tailweight = Parameter(self._tailweight)

    def forward(self, x):
        return torch.sinh((utils.arcsinh(x) + self.skewness) * self.tailweight)

    def inverse(self, y):
        return torch.sinh(utils.arcsinh(y) / self.tailweight - self.skewness)

    def log_abs_det_jacobian(self, x, y):
        return (torch.log(
                torch.cosh((utils.arcsinh(x) + self.skewness) * self.tailweight)
                / utils.sqrtx2p1(x + 1e-10)) + torch.log(self.tailweight))

    @property
    def tailweight(self):
        return softplus(self._tailweight)

    def get_parameters(self):
        return {'type':'sinharcsinh', 'skewness':self.skewness.detach().numpy(),
                'tailweight':self.tailweight.detach().numpy()}
