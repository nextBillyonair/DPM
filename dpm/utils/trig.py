import torch
from .constants import pi

# regular

def sin(x):
    return torch.sin(x)

def cos(x):
    return torch.cos(x)

def tan(x):
    return torch.tan(x)

def cot(x):
    return 1. / torch.tan(x)

def sec(x):
    return 1. / torch.cos(x)

def csc(x):
    return 1. / torch.sin(x)

# regular inverse

def arcsin(x):
    return x.asin()

def arccos(x):
    return x.acos()

def arctan(x):
    return x.atan()

def arccot(x):
    return (pi / 2.) - arctan(x)

def arcsec(x):
    return arccos(1. / x)

def arccsc(x):
    return arcsin(1. / x)


# Hyperbolic

def sinh(x):
    return torch.sinh(x)

def cosh(x):
    return torch.cosh(x)

def tanh(x):
    return torch.tanh(x)

def coth(x):
    return torch.cosh(x) / torch.sinh(x)

def sech(x):
    return 1. / torch.cosh(x)

def csch(x):
    return 1. / torch.sinh(x)


# Hyperbolic Inverse

def arcsinh(x):
    return torch.log(x + (x.pow(2) + 1).sqrt())

def arccosh(x):
    return torch.log(x + (x.pow(2) - 1).sqrt())

def arctanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

def arccoth(x):
    return 0.5 * (torch.log(x + 1) - torch.log(x - 1))

def arcsech(x):
    return (1 + (1 - x.pow(2)).sqrt()).log() - x.log()

def arccsch(x):
    return ((1. / x) + ((1. / x.pow(2)) - 1).sqrt()).log()
