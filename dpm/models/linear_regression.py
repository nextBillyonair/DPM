import torch
from functools import partial
from dpm.distributions import Distribution, Normal, ConditionalModel, Laplace
from .model import Model

class LinearRegression(Model):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=[], activation="",
                            output_shapes=[output_shape],
                            output_activations=[None],
                            distribution=partial(Normal, scale=torch.ones(1, output_shape)))

    def log_prob(self, x, y):
        return self.model.log_prob(y, x)

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)


class L1Regression(Model):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=[], activation="",
                            output_shapes=[output_shape],
                            output_activations=[None],
                            distribution=partial(Laplace, scale=torch.ones(output_shape)))


    def log_prob(self, x, y):
        return self.model.log_prob(y, x)

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)


class RidgeRegression(Model):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=[], activation="",
                            output_shapes=[output_shape],
                            output_activations=[None],
                            distribution=partial(Normal, scale=torch.ones(1, output_shape)))
        self.prior = Normal(0., tau)

    def prior_penalty(self):
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.model.parameters()])).sum()

    def log_prob(self, x, y):
        return self.model.log_prob(y, x) + self.prior_penalty()

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)


class LassoRegression(Model):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=[], activation="",
                            output_shapes=[output_shape],
                            output_activations=[None],
                            distribution=partial(Normal, scale=torch.ones(1, output_shape)))
        self.prior = Laplace(0., tau)

    def prior_penalty(self):
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.model.parameters()])).sum()

    def log_prob(self, x, y):
        return self.model.log_prob(y, x) + self.prior_penalty()

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)
