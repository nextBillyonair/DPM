from torch.nn import ModuleList
from .transform import Transform


# Try to use this in transformed dist
class Chain(Transform):

    def __init__(self, transforms):
        super().__init__()
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.chain = ModuleList(transforms)

    def forward(self, x):
        for transform in self.chain:
            x = transform.forward(x)
        return x

    def inverse(self, y):
        for transform in reversed(self.chain):
            y = transform.inverse(y)
        return y

    def log_abs_det_jacobian(self, x, y=None):
        prev_value = x
        log_det = 0.0
        for transform in reversed(self.chain):
            value = transform.inverse(prev_value)
            log_det += transform.log_abs_det_jacobian(value, prev_value)
            prev_value = value
        return log_det, value

    def get_parameters(self):
        return {'type':'chain',
                'compnents':[transform.get_parameters() for transform in self.chain]}

    def __iter__(self):
        return iter(self.chain)

    def __getitem__(self, index):
        return self.chain[index]
