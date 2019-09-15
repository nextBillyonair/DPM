from .transform import Transform
import torch


class Softsign(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1.0 + x.abs())

    def inverse(self, y):
        return y / (1.0 - y.abs())

    def log_abs_det_jacobian(self, x, y):
        return (2.0 * torch.log1p(-torch.abs(y))).sum(-1)

    def get_parameters(self):
        return {'type':'softsign'}
