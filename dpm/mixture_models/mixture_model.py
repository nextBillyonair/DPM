import torch
from torch.distributions import Categorical
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
import numpy as np

from dpm.distributions import Distribution


# Non-differentiable Categorical weights (not learnable)
class MixtureModel(Distribution):
    def __init__(self, models, probs):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.categorical = Categorical(probs=torch.tensor(probs))
        self.models = ModuleList(models)

    def log_prob(self, value):
        log_probs = torch.stack([sub_model.log_prob(value)
                                 for sub_model in self.models])
        cat_log_probs = self.categorical.probs.view(-1, 1).log()
        return torch.logsumexp(log_probs + cat_log_probs, dim=0)

    def sample(self, batch_size):
        indices = self.categorical.sample((batch_size,))
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models])
        return samples[indices, np.arange(batch_size)]

    def get_parameters(self):
        return {'probs': self.categorical.probs.numpy(),
                'models': [model.get_parameters() for model in self.models]}
