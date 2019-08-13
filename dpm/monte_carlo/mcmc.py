import torch
from functools import partial
from dpm.distributions import Normal, Distribution, Langevin
from dpm.newton import gradient


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


def grad_U(q, model):
    q.requires_grad = True
    g = gradient(-model.log_prob(q), q).squeeze(0).detach()
    q.requires_grad = False
    return g


def hamiltonian_monte_carlo(model, epsilon=0.2, leapfrog=20, alpha=1.,
                            epochs=1000, burn_in=1000, keep_every=1, init=None):
    if init is None:
        current_q = torch.rand((1, model.n_dims))
    else:
        current_q = init
        if not isinstance(current_q, torch.Tensor):
            current_q = torch.tensor(current_q).view(1, -1)

    samples = []

    for t in range(epochs):
        q = current_q.clone()
        p = torch.randn((q.size(1), 1))
        current_p = p.clone()

        p = p - epsilon * grad_U(q, model) / 2.

        for l in range(leapfrog):
            q = q + epsilon * p
            if l + 1 != leapfrog:
                tampering = alpha if l < (leapfrog / 2.) else (1. / alpha)
                p = tampering * (p - epsilon * grad_U(q, model))

        p = p - epsilon * grad_U(q, model) / 2.
        p = -p

        current_U = -model.log_prob(current_q)
        current_K = current_p.pow(2).sum() / 2.
        proposed_U = -model.log_prob(q)
        proposed_K = p.pow(2).sum() / 2.

        u = torch.rand(1).log()
        A = current_U - proposed_U + current_K - proposed_K
        if u < A:
            current_q = q

        if t >= burn_in and t % keep_every == 0:
            samples.append(current_q)

    return torch.cat(samples, dim=0)












#EOF
