from .transform import Transform
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch
import dpm.utils as utils


class Kumaraswamy(Transform):

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

    def forward(self, x):
        return ((-((-x).log1p() / self.beta).exp()).log1p() / self.alpha).exp()

    def inverse(self, y):
        return (-(1 - y.pow(self.alpha)).pow(self.beta)).log1p().exp()

    def log_abs_det_jacobian(self, x, y):
        return (-(self.alpha.log() + self.beta.log() \
            + (self.alpha - 1) * utils.log(y) + (self.beta - 1) * (-y.pow(self.alpha)).log1p())).sum(-1)

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        return {'type':'kumaraswamy',
                'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}
