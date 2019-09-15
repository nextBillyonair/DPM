from .transform import Transform
from torch.nn import Parameter
import torch


class Affine(Transform):

    def __init__(self, loc=0.0, scale=1.0, learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(1, -1)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(1, -1)
        self.loc = loc.float()
        self.scale = scale.float()
        self.n_dims = len(loc)
        if learnable:
            self.loc = Parameter(self.loc)
            self.scale = Parameter(self.scale)

    def forward(self, x):
        return self.loc + self.scale * x

    def inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        return torch.log(torch.abs(self.scale.expand(x.size()))).sum(-1)

    def get_parameters(self):
        return {'type':'affine', 'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}
