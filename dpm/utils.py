import torch


def asinh(x):
    return torch.log(x + (x.pow(2) + 1).pow(0.5))

def atanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

def sqrtx2p1(x):
    return x.abs() * (1 + x.pow(-2)).sqrt()

def softplus_inverse(value, threshold=20):
    inv = torch.log((value.exp() - 1.0))
    inv[value > threshold] = value[value > threshold]
    return inv
