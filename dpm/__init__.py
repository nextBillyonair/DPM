import numpy as np
from torch import nn

def num_parameters(self):
    return sum(np.prod(p.shape) for p in self.parameters())

def named_parameters_list(self):
    return [p for p in self.named_parameters()]

nn.Module.num_parameters = property(num_parameters)
nn.Module.named_parameters_list = property(named_parameters_list)
