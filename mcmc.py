import torch
from torch.distributions import MultivariateNormal


def metropolis_hastings(true_model, epochs=10000, burn_in=1000,
                        keep = 200, variance=None):
    x_t = torch.zeros((1, true_model.n_dims))
    if variance is None:
        variance = torch.eye(true_model.n_dims)
    t = 0
    samples = [x_t]
    while t < epochs:
        model = MultivariateNormal(x_t.squeeze(0), variance)
        x_prime = model.sample((1,))
        p_prob = true_model.log_prob(x_prime) - true_model.log_prob(x_t)
        A = min(0, p_prob)
        u = torch.log(torch.rand(1))
        if u <= A :
            x_t = x_prime
            if t >= burn_in:
                samples.append(x_t)
        elif keep is not None and t % keep == 0:
            if t >= burn_in:
                samples.append(x_t)
        t += 1
    return torch.stack(samples).squeeze(1)
