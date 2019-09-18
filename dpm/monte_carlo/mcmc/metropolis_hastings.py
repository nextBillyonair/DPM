import torch
from functools import partial
from dpm.distributions import Normal, Langevin


def metropolis(true_model, epochs=10000, burn_in=1000, keep_every=1,
               variance=None, init=None):
    if variance is None:
        variance = torch.eye(true_model.n_dims)
    proposal_model = partial(Normal, scale=variance, learnable=False)
    return metropolis_hastings(true_model, proposal_model, epochs, burn_in,
                               keep_every, init)


def metropolis_langevin(true_model, tau=1., epochs=10000,
                        burn_in=1000, keep_every=1, init=None):
    langevin_model = partial(Langevin, model=true_model, tau=tau)
    return metropolis_hastings(true_model, langevin_model, epochs, burn_in,
                               keep_every, init)


def metropolis_hastings(true_model, proposal_model, epochs=10000, burn_in=1000,
                        keep_every=1, init=None):
    if init is None:
        x_t = torch.rand((1, true_model.n_dims))
    else:
        x_t = init
        if not isinstance(x_t, torch.Tensor):
            x_t = torch.tensor(x_t).view(1, -1)

    samples = []
    for t in range(epochs):
        model = proposal_model(x_t.squeeze(0))
        x_prime = model.sample(1)
        p_prob = (true_model.log_prob(x_prime) - true_model.log_prob(x_t)) \
                 + (proposal_model(x_prime.squeeze(0)).log_prob(x_t) \
                 - model.log_prob(x_prime))
        A = min(0, p_prob)
        u = torch.rand(1).log()
        if u <= A :
            x_t = x_prime

        if t >= burn_in and t % keep_every == 0:
            samples.append(x_t)

    return torch.cat(samples, dim=0)
