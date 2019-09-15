from .transform import Transform
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch
import dpm.utils as utils

class Gumbel(Transform):

    def __init__(self, loc=0.0, scale=1.0, learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(1, -1)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(1, -1)
        self.loc = loc.float()
        self._scale = utils.softplus_inverse(scale.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def forward(self, x):
        z = (x - self.loc) / self.scale
        return torch.exp(-torch.exp(-z))

    def inverse(self, y):
        return self.loc - self.scale * torch.log(-torch.log(y))

    def log_abs_det_jacobian(self, x, y):
        return -torch.log(self.scale / (-torch.log(y) * y)).sum(-1)

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        return {'type':'gumbel', 'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
