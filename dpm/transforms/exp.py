from .transform import Transform


class Exp(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp()

    def inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x

    def get_parameters(self):
        return {'type':'exp'}
