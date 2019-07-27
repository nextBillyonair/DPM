from .transform import Transform
import torch
from torch.nn.functional import softplus
import dpm.utils as utils


class Sigmoid(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def inverse(self, y):
        return utils.logit(y)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x) - softplus(x)

    def get_parameters(self):
        return {'type':'sigmoid'}
