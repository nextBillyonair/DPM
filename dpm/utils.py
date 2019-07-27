import torch

# constants

euler_mascheroni = 0.57721566490153286060651209008240243104215933593992


#functions

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

def sqrtx2p1(x):
    return x.abs() * (1 + x.pow(-2)).sqrt()

def softplus_inverse(value, threshold=20):
    inv = torch.log((value.exp() - 1.0))
    inv[value > threshold] = value[value > threshold]
    return inv

def logit(x):
    return x.log() - (-x).log1p()
