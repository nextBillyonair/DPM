import torch


def atanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


def softplus_inverse(value, threshold=20):
    inv = torch.log((value.exp() - 1.0))
    inv[value > threshold] = value[value > threshold]
    return inv
