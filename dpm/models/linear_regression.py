import torch
from torch.nn import Softmax
from functools import partial
from dpm.distributions import (
    Distribution, Normal, ConditionalModel, Laplace,
    Bernoulli, Categorical
)
from dpm.utils import Sigmoid
from .model import Model

class LinearModel(Model):

    def __init__(self, input_dim=1, output_shape=1, output_activation=None,
                 distribution=None, prior=None):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=[], activation="",
                            output_shapes=[output_shape],
                            output_activations=[output_activation],
                            distribution=distribution)
        self.prior = prior

    def prior_penalty(self):
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.model.parameters()]).view(-1, 1)).sum()

    def log_prob(self, x, y):
        if self.prior:
            return self.model.log_prob(y, x) + self.prior_penalty()
        return self.model.log_prob(y, x)

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)


class LinearRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)))


class L1Regression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape,
            distribution=partial(Laplace, scale=torch.ones(output_shape)))


class RidgeRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)),
            prior=Normal(0., tau))


class LassoRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)),
            prior=Laplace(0., tau))


class LogisticRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape, output_activation=Sigmoid(),
            distribution=Bernoulli)


# multiclass, max ent
class SoftmaxRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=2):
        super().__init__(input_dim, output_shape, output_activation=Softmax(dim=1),
            distribution=Categorical)




# EOF