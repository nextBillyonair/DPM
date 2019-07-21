import torch
from torch.nn import Module
from abc import abstractmethod, ABC
from dpm.mixture_models import MixtureModel

def cross_entropy(p_model, q_model, batch_size=64):
    return p_model.cross_entropy(q_model, batch_size)


def perplexity(p_model, q_model, batch_size=64):
    return cross_entropy(p_model, q_model, batch_size).exp()


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



################################################################################
# Experimental


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


def exponential_divergence(p_model, q_model, batch_size=64):
    p_samples = p_model.sample(batch_size)
    return (p_model.log_prob(p_samples) - q_model.log_prob(p_samples)).pow(2).mean()

class FDivergence(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def f_div(self, t):
        raise NotImplementedError()

    def forward(self, p_model, q_model, batch_size=64):
        q_samples = q_model.sample(batch_size)
        odds_ratio = (p_model.log_prob(q_samples) - q_model.log_prob(q_samples)).exp()
        # q_prob = q_model.log_prob(q_samples).exp()
        # return (q_prob * self.f_div(odds_ratio)).sum(-1)
        return self.f_div(odds_ratio).mean()


class ForwardKL(FDivergence):

    def f_div(self, t):
        return t * (t + 1e-10).log()

class ReverseKL(FDivergence):

    def f_div(self, t):
        return -(t + 1e-10).log()

class TotalVariation(FDivergence):

    def f_div(self, t):
        return 0.5 * (t - 1.).abs()

class HellingerDistance(FDivergence):

    def f_div(self, t):
        return (t.sqrt() - 1.).pow(2)

class Pearson(FDivergence):

    def f_div(self, t):
        return (t - 1.).pow(2)

class Neyman(FDivergence):

    def f_div(self, t):
        return (1 / t) - 1.


class Alpha(FDivergence):

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = 1

    def f_div(self, t):
        if self.alpha == -1:
            return -t.log()
        if self.alpha == 1:
            return t * t.log()
        return (4 / (1-self.alpha**2)) * (1 - t.pow((1+self.alpha)/2))


# EOF
