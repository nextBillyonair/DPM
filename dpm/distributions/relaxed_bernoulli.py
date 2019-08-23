import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class RelaxedBernoulli(Distribution):

    def __init__(self, probs=[0.5], temperature=1.0, learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = torch.tensor(temperature)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = utils.softplus_inverse(probs.float())
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = dists.RelaxedBernoulli(self.temperature, self.probs)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = dists.RelaxedBernoulli(self.temperature, self.probs)
        return model.sample((batch_size,))

    @property
    def probs(self):
        return softplus(self.logits)

    def get_parameters(self):
        if self.n_dims == 1: return {'probs':self.probs.item()}
        return {'probs':self.probs.detach().numpy()}
