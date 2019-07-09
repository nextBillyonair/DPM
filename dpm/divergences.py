import torch
from torch.nn import Module
from dpm.mixture_models import MixtureModel


def cross_entropy(p_model, q_model, batch_size=64):
    return p_model.cross_entropy(q_model, batch_size)

def forward_kl(p_model, q_model, batch_size=64):
    p_samples = p_model.sample(batch_size)
    return (p_model.log_prob(p_samples) - q_model.log_prob(p_samples)).mean()

def reverse_kl(p_model, q_model, batch_size=64):
    q_samples = q_model.sample(batch_size)
    return -(p_model.log_prob(q_samples) - q_model.log_prob(q_samples)).mean()


def js_divergence(p_model, q_model, batch_size=64):
    M = MixtureModel([p_model, q_model], [0.5, 0.5])
    return 0.5 * (forward_kl(p_model, M, batch_size)
                  + forward_kl(q_model, M, batch_size))


def total_variation(p_model, q_model, batch_size=64):
    mixture_model = MixtureModel([p_model, q_model], [0.5, 0.5])
    samples = mixture_model.sample(batch_size)
    ratio = 0.5 * ((p_model.log_prob(samples) - mixture_model.log_prob(samples)).exp() \
             - (q_model.log_prob(samples) - mixture_model.log_prob(samples)).exp()).abs()
    return ratio.mean()


def pearson(p_model, q_model, batch_size=64):
    mixture_model = MixtureModel([p_model, q_model], [0.5, 0.5])
    samples = mixture_model.sample(batch_size)
    ratio = ((p_model.log_prob(samples) - mixture_model.log_prob(samples)).exp() \
             - (q_model.log_prob(samples) - mixture_model.log_prob(samples)).exp()).pow(2)
    return ratio.mean()


def _other_term(p_model, q_model, batch_size):
    p_samples = p_model.sample(batch_size)
    p_log_prob = p_model.log_prob(p_samples)
    q_log_prob = q_model.log_prob(p_samples)
    return (torch.logsumexp(torch.stack([p_log_prob, q_log_prob]), dim=0) - torch.logsumexp(torch.stack([p_log_prob, q_log_prob, q_log_prob]), dim=0)).mean()

def js_divergence_2(p_model, q_model, batch_size=64):
    M = MixtureModel([p_model, q_model], [0.5, 0.5])
    return 0.5 * (_other_term(p_model, M, batch_size)
                  + _forward_kl(q_model, M, batch_size))






# EOF
