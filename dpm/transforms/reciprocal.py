from .transform import Transform


class Reciprocal(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1. / x

    def inverse(self, y):
        return 1. / y

    def log_abs_det_jacobian(self, x, y):
        return (-2. * x.abs().log()).sum(-1)

    def get_parameters(self):
        return {'type':'reciprocal'}
