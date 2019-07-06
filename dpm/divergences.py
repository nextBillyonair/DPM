import torch
from torch.nn import Module
from dpm.mixture_models import MixtureModel


def forward_kl(p_model, q_model, batch_size=64):
    p_samples = p_model.sample(batch_size)
    return -(q_model.log_prob(p_samples)).mean()

def reverse_kl(p_model, q_model, batch_size=64):
    q_samples = q_model.sample(batch_size)
    return -(p_model.log_prob(q_samples) - q_model.log_prob(q_samples)).mean()

def _forward_kl(p_model, q_model, batch_size=64):
    p_samples = p_model.sample(batch_size)
    return (p_model.log_prob(p_samples) - q_model.log_prob(p_samples)).mean()

def js_divergence(p_model, q_model, batch_size=64):
    M = MixtureModel([p_model, q_model], [0.5, 0.5])
    return 0.5 * (_forward_kl(p_model, M, batch_size)
                  + _forward_kl(q_model, M, batch_size))
