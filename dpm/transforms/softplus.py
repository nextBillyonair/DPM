from .transform import Transform
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch
import dpm.utils as utils

class Softplus(Transform):

    def __init__(self, hinge_softness=1.0, learnable=True):
        super().__init__()
        if hinge_softness == 0.0: raise ValueError("Hinge Softness cannot be 0")
        if not isinstance(hinge_softness, torch.Tensor):
            hinge_softness = torch.tensor(hinge_softness).view(1, -1)
        self.hinge_softness = hinge_softness
        if learnable:
            self.hinge_softness = Parameter(self.hinge_softness)

    def forward(self, x):
        return self.hinge_softness * softplus(x / self.hinge_softness)

    def inverse(self, y):
        return self.hinge_softness * utils.softplus_inverse(y / self.hinge_softness)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x / self.hinge_softness)
        # return torch.log(-torch.expm1(-y) + 1e-10)

    def get_parameters(self):
        return {'type':'softplus',
                'hinge_softness':self.hinge_softness.detach().numpy()}
