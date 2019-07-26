import torch
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
import numpy as np

from dpm.distributions import Distribution, GumbelSoftmax



# Differentiable, Learnable Mixture Weights
class GumbelMixtureModel(Distribution):
    def __init__(self, models, probs, temperature=1.0, hard=True):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.categorical = GumbelSoftmax(probs, temperature, hard)
        self.models = ModuleList(models)

    def log_prob(self, value):
        log_probs = torch.stack([sub_model.log_prob(value)
                                 for sub_model in self.models])
        cat_log_probs = self.categorical.probs.view(-1, 1).log()
        return torch.logsumexp(log_probs + cat_log_probs, dim=0)

    def sample(self, batch_size):
        one_hot = self.categorical.sample(batch_size)
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models], dim=1)
        return (samples * one_hot.unsqueeze(2)).sum(dim=1)

    def get_parameters(self):
        return {'probs': self.categorical.probs.detach().numpy(),
                'models': [model.get_parameters() for model in self.models]}
