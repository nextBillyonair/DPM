import numpy as np
from torch import nn

def num_parameters(self):
    return sum(np.prod(p.shape) for p in self.parameters())

nn.Module.num_parameters = property(num_parameters)
