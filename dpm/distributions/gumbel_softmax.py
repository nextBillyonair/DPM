import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class GumbelSoftmax(Distribution):

    def __init__(self, probs=[0.5, 0.5], temperature=1.0,
                 hard=True, learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = temperature
        self.hard = hard
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = probs.log()
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = dists.Categorical(probs=self.probs)
        return model.log_prob(value)

    def sample(self, batch_size):
        U = torch.rand((batch_size, self.n_dims))
        gumbel_samples = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = self.logits + gumbel_samples
        y = (y / self.temperature).softmax(dim=1)
        if self.hard:
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y = (y_hard - y).detach() + y
        return y

    @property
    def probs(self):
        return self.logits.softmax(dim=-1)

    def get_parameters(self):
        return {'probs':self.probs.detach().numpy()}