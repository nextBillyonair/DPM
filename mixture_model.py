import torch
from torch.distributions import Categorical
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
from torch.distributions import Normal
import numpy as np

from distributions import Distribution, GumbelSoftmax

# Non-differentiable Categorical weights (not learnable)
class MixtureModel(Distribution):
    def __init__(self, models, probs):
        super(MixtureModel, self).__init__()
        self.n_dims = models[0].n_dims
        self.categorical = Categorical(probs=torch.tensor(probs))
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

    def get_parameters(self):
        return {'probs': self.categorical.probs.numpy(),
                'models': [model.get_parameters() for model in self.models]}


# Differentiable, Learnable Mixture Weights
class GumbelMixtureModel(Distribution):
    def __init__(self, models, probs, temperature=1.0, hard=True):
        super(GumbelMixtureModel, self).__init__()
        self.n_dims = models[0].n_dims
        self.categorical = GumbelSoftmax(probs, temperature, hard)
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

    def get_parameters(self):
        return {'probs': self.categorical.probs.detach().numpy(),
                'models': [model.get_parameters() for model in self.models]}


class StudentTMixtureModel(Distribution):
    def __init__(self, loc, scale, df):
        self.n_dims = len(loc)
        self.loc = Parameter(torch.tensor(loc))
        self._scale = Parameter(self.softplus_inverse(torch.tensor(scale)))
        self._df = Parameter(self.softplus_inverse(torch.tensor(df)))

    def log_prob(self, value):
        pass

    def sample(self, batch_size):
        pass

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def df(self):
        return softplus(self._df)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc' : self.loc.item(), 'scale':self.scale.item(),
                    'df':self.df.item()}
        return {'loc' : self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy(),
                'df':self.df.detach().numpy()}






# EOF
