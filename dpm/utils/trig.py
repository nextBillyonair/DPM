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


# Versine

def versin(x):
    return 1 - cos(x)

def vercos(x):
    return 1 + cos(x)

def coversin(x):
    return 1 - sin(x)

def covercos(x):
    return 1 + sin(x)

def haversin(x):
    return versin(x) / 2

def havercos(x):
    return vercos(x) / 2

def hacoversin(x):
    return coversin(x) / 2

def hacovercos(x):
    return covercos(x) / 2


def arcversin(x):
    return arccos(1 - x)

def arcvercos(x):
    return arccos(1 + x)

def arccoversin(x):
    return arcsin(1 - x)

def arccovercos(x):
    return arcsin(1 + x)

def archaversin(x):
    return 2 * arcsin(x.sqrt())

def archavercos(x):
    return 2 * arccos(x.sqrt())




# EOF
