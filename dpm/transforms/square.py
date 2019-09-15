from .transform import Transform
import numpy as np


class Square(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.pow(2)

    def inverse(self, y):
        return y.sqrt()

    def log_abs_det_jacobian(self, x, y):
        return (np.log(2.) + x.log()).sum(-1)

    def get_parameters(self):
        return {'type':'square'}
