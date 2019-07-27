from .transform import Transform


class InverseTransform(Transform):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform.inverse(x)

    def inverse(self, y):
        return self.transform(y)

    def log_abs_det_jacobian(self, x, y):
        return -self.transform.log_abs_det_jacobian(y, x)

    def get_parameters(self):
        return {'type':'inverse'}
