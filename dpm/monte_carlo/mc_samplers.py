import torch
from dpm.distributions import Beta, Uniform
import numpy as np
import math

# SAMPLING

def lcg(modulus=2147483563, a=40014, c=0, seed=42):
    while True:
        seed = (a * seed + c) % modulus
        yield seed

def rand(batch_size=10000, generator=None):
    if generator is None:
        generator = lcg()
    samples = torch.tensor([[next(generator)] for _ in range(batch_size)]).float()
    return samples / 2147483563.


# function is the inverse CDF
def inverse_sampling(function, batch_size=10000, n_dims=1):
    U = torch.rand((batch_size, n_dims))
    return function(U)


def rejection_sampling(model, test_model, M, batch_size=10000):
    # if M <= 1: raise ValueError(f'Error: M should be larger than 1; got {M}')
    model_samples = test_model.sample(batch_size)
    uniform_samples = torch.rand(batch_size).log()
    acceptance_ratio = (model.log_prob(model_samples) \
                       - test_model.log_prob(model_samples) \
                       - np.log(M))
    accepted = uniform_samples < acceptance_ratio
    return model_samples[accepted]

def box_muller(batch_size=10000):
    U1 = torch.rand((batch_size, 1))
    U2 = torch.rand((batch_size, 1))
    R, V = -2 * U1.log(), 2 * np.pi * U2
    Z1 = R.sqrt() * torch.cos(V)
    Z2 = R.sqrt() * torch.sin(V)
    return Z1, Z2

def marsaglia_bray(batch_size=10000):
    U1 = torch.rand((batch_size, 1))
    U2 = torch.rand((batch_size, 1))
    U1, U2 = 2 * U1 - 1, 2 * U2 - 1
    X = U1.pow(2) + U2.pow(2)
    Y = torch.sqrt(-2 * X.log() / X)
    Z1, Z2 = U1 * Y, U2 * Y
    Z1 = Z1[X <= 1].reshape(-1, 1)
    Z2 = Z2[X <= 1].reshape(-1, 1)
    return Z1, Z2

# automate
def mode_sampling(model, rng=(-10, 10), batch_size=10000):
    c = model.log_prob(model.mode.view(-1, 1))
    if torch.isnan(c).item() or torch.isinf(c).item():
        c = torch.tensor([0.])
    U1 = Uniform(rng[0], rng[1], learnable=False).sample(batch_size)
    U2 = torch.rand((batch_size, 1))
    accepted = c + U2.log() <= model.log_prob(U1).view(-1, 1)
    return U1[accepted].reshape(-1, model.n_dims)

# Only for a, b > 1 for now.
def beta_sampling(alpha, beta, batch_size=10000):
    f = Beta(alpha, beta, learnable=False)
    return mode_sampling(f, rng=(0, 1), batch_size=batch_size)

# possible broken for values, dowsn't work for n_dims > 2
def double_exponential(batch_size=10000, n_dims=1):
    X = -torch.rand((batch_size, n_dims)).log()
    U2 = torch.rand((batch_size, n_dims))
    U3 = torch.rand((batch_size, n_dims))
    X[U3 <= 0.5] = -X[U3 <= 0.5]
    threshold = (-0.5 * ((X - 1).pow(2))).exp()
    return X[U2 > threshold].reshape(-1, n_dims)
