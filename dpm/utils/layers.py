import torch
from torch.nn import Module
from torch.nn.functional import softplus
from .math import logit

# Layers

# Converts function to torch nn Module (for sequential)
class Function(Module):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


# Avoids NAN
class SafeSoftplus(Function):

    def __init__(self):
        super().__init__(lambda x : softplus(x) + 0.01)


# Avoids torch deprecation warning
class Sigmoid(Function):

    def __init__(self):
        super().__init__(torch.sigmoid)


class Logit(Function):

    def __init__(self):
        super().__init__(logit)



class Flatten(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)



# EOF
