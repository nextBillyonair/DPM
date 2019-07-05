import torch
from torch.nn import Module
from mixture_model import MixtureModel

class ForwardKL(Module):
    def __init__(self, n_dims):
        super().__init__()

    def forward(self, p_model, q_model, batch_size=64):
        p_samples = p_model.sample(batch_size)
        return -(q_model.log_prob(p_samples)).mean()


class ReverseKL(Module):
    def __init__(self, n_dims):
        super().__init__()

    def forward(self, p_model, q_model, batch_size=64):
        q_samples = q_model.sample(batch_size)
        return -(p_model.log_prob(q_samples) - q_model.log_prob(q_samples)).mean()


class JSDivergence(Module):
    def __init__(self, n_dims):
        super().__init__()

    def _forward_kl(self, p_model, q_model, batch_size=64):
        p_samples = p_model.sample(batch_size)
        return (p_model.log_prob(p_samples) - q_model.log_prob(p_samples)).mean()

    def forward(self, p_model, q_model, batch_size=64):
        M = MixtureModel([p_model, q_model], [0.5, 0.5])
        return 0.5 * (self._forward_kl(p_model, M, batch_size)
                      + self._forward_kl(q_model, M, batch_size))
