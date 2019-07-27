from .transform import Transform


class Expm1(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.expm1()

    def inverse(self, y):
        return y.log1p()

    def log_abs_det_jacobian(self, x, y):
        # log1p(y) = log1p(e^x - 1) = log((e^x - 1) + 1) = x
        return x

    def get_parameters(self):
        return {'type':'expm1'}
