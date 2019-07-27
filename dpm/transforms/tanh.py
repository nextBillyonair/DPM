from .transform import Transform
import dpm.utils as utils
import numpy as np
from torch.nn.functional import softplus

class Tanh(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()

    def inverse(self, y):
        return utils.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2.0) - x - softplus(-2.0 * x))

    def get_parameters(self):
        return {'type':'tanh'}
