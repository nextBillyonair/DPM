from .inverse_transform import InverseTransform
from .sigmoid import Sigmoid


class Logit(InverseTransform):

    def __init__(self):
        super().__init__(Sigmoid())

    def get_parameters(self):
        return {'type':'logit'}
