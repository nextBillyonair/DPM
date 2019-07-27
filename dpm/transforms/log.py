from .inverse_transform import InverseTransform
from.exp import Exp

class Log(InverseTransform):

    def __init__(self):
        super().__init__(Exp())

    def get_parameters(self):
        return {'type':'log'}
