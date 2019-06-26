import torch
from torch.distributions import Categorical
from torch.nn import Module, ModuleList
import numpy as np

from distributions import Distribution, GumbelSoftmax

# Non-differentiable Categorical weights (not learnable)
class MixtureModel(Distribution):
    def __init__(self, models, weights):
        super(MixtureModel, self).__init__()
        self.categorical = Categorical(probs=torch.tensor(weights))
        self.models = ModuleList(models)

    def log_prob(self, value):
        log_probs = torch.stack([sub_model.log_prob(value)
                                 for sub_model in self.models])
        return torch.logsumexp(log_probs + self.categorical.probs.unsqueeze(1).log(), dim=0)

    def sample(self, batch_size):
        indices = self.categorical.sample((batch_size,))
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models])
        return samples[indices, np.arange(batch_size)]

    @property
    def n_dims(self):
        return self.models[0].n_dims


# Differentiable, Learnable Mixture Weights
class GumbelMixtureModel(Distribution):
    def __init__(self, models, weights, temperature=1.0, hard=True):
        super(GumbelMixtureModel, self).__init__()
        self.categorical = GumbelSoftmax(weights, temperature, hard)
        self.models = ModuleList(models)

    def log_prob(self, value):
        log_probs = torch.stack([sub_model.log_prob(value)
                                 for sub_model in self.models])
        return torch.logsumexp(log_probs + self.categorical.probs.unsqueeze(1).log(), dim=0)

    def sample(self, batch_size):
        one_hot = self.categorical.sample(batch_size)
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models], dim=1)
        return (samples * one_hot.unsqueeze(2)).sum(dim=1)

    @property
    def n_dims(self):
        return self.models[0].n_dims
