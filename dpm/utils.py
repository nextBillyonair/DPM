import torch
import numpy as np


def histogram(tensor, bins=10):
    return np.histogram(tensor.reshape(-1).numpy(), bins=bins)






# EOF
