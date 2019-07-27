from .transform import Transform
import torch
from torch.nn.functional import softplus


class Sigmoid(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def inverse(self, y):
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x) - softplus(x)
        # return -torch.log((y.reciprocal() + (1 - y).reciprocal()))

    def get_parameters(self):
        return {'type':'sigmoid'}
