import torch
from torch.distributions import Beta
import numpy as np
import math

# SAMPLING

# function is the inverse CDF
def inverse_sampling(function, batch_size=10000, n_dims=1):
    U = torch.rand((batch_size, n_dims))
    return function(U)


def rejection_sampling(model, test_model, M, batch_size=10000):
    if M <= 1: raise ValueError(f'Error: M should be larger than 1; got {M}')
    model_samples = test_model.sample(batch_size)
    uniform_samples = torch.rand(batch_size).log()
    acceptance_ratio = (model.log_prob(model_samples) \
                       - test_model.log_prob(model_samples) \
                       - np.log(M))
    accepted = uniform_samples < acceptance_ratio
    return model_samples[accepted]

# Works!
def box_muller(batch_size=10000, n_dims=1):
    U1 = torch.rand((batch_size, n_dims))
    U2 = torch.rand((batch_size, n_dims))
    R, V = -3 * U1.log(), 2 * math.pi * U2
    Z1 = R.sqrt() * torch.cos(V)
    Z2 = R.sqrt() * torch.sin(V)
    return torch.cat((Z1, Z2))

# Works!
def marsaglia_bray(batch_size=10000, n_dims=1):
    U1 = torch.rand((batch_size, n_dims))
    U2 = torch.rand((batch_size, n_dims))
    U1, U2 = 2 * U1 - 1, 2 * U2 - 1
    X = U1.pow(2) + U2.pow(2)
    Y = torch.sqrt(-2 * X.log() / X)
    Z1, Z2 = U1 * Y, U2 * Y
    Z1 = Z1[X <= 1].reshape(-1, n_dims)
    Z2 = Z2[X <= 1].reshape(-1, n_dims)
    return torch.cat((Z1, Z2))


# possible broken for values, dowsn't work for n_dims > 2
def beta_sampling(alpha, beta, batch_size=10000, n_dims=1):
    f = Beta(alpha, beta)
    c = (alpha - 1) / (alpha + beta - 2)
    U1 = torch.rand((batch_size, n_dims))
    U2 = torch.rand((batch_size, 1))
    accepted = c + U2.log() <= f.log_prob(U1)
    return U1[accepted].reshape(-1, n_dims)

# possible broken for values, dowsn't work for n_dims > 2
def double_exponential(batch_size=10000, n_dims=1):
    X = -torch.rand((batch_size, n_dims)).log()
    U2 = torch.rand((batch_size, n_dims))
    U3 = torch.rand((batch_size, n_dims))
    X[U3 <= 0.5] = -X[U3 <= 0.5]
    threshold = (-0.5 * ((X - 1).pow(2))).exp()
    return X[U2 > threshold].reshape(-1, n_dims)