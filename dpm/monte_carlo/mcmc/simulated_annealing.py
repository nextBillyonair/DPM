import torch
import math
from functools import partial
from dpm.distributions import Normal


def logarithmic_annealing(t, k, t_0, gamma):
    return t_0 * math.log(2) / math.log(k + 2)

def exponentional_annealing(t, k, t_0, gamma):
    return gamma * t

def fast_annealing(t, k, t_0, gamma):
    return t_0 / (k + 1)


def simulated_annealing(true_model, proposal_model=None, annealing_scheduler=None,
                        initial_temperature=10.0, gamma=0.5,
                        epochs=10000, burn_in=1000, keep_every=1, init=None):
    if init is None:
        x_k = torch.rand((1, true_model.n_dims))
    else:
        x_k = init
        if not isinstance(x_t, torch.Tensor):
            x_k = torch.tensor(x_t).view(1, -1)

    if proposal_model is None:
        proposal_model = partial(Normal, scale=1.0, learnable=False)

    if annealing_scheduler is None:
        annealing_scheduler = lambda t, k, t_0, g : t

    samples = []
    t = initial_temperature

    for k in range(epochs):
        model = proposal_model(x_k.squeeze(0))
        x_prime = model.sample(1)
        diff = true_model.log_prob(x_k) - true_model.log_prob(x_prime)

        if diff <= 0 or torch.rand(1) < (-diff / t).exp():
            x_k = x_prime

        t = annealing_scheduler(t, k, initial_temperature, gamma)

        if k >= burn_in and k % keep_every == 0:
            samples.append(x_k)

    return torch.cat(samples, dim=0)
