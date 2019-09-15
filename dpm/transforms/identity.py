from .transform import Transform
import torch

class Identity(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x).sum(-1)

    def get_parameters(self):
        return {'type':'identity'}
