import torch
from torch.distributions import Categorical
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
import numpy as np

from dpm.distributions import Distribution, GumbelSoftmax, Gamma, Normal

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


# Infinite Mixture Version of Gaussian
class InfiniteMixtureModel(Distribution):
    def __init__(self, df, loc, scale,
                 loc_learnable=True,
                 scale_learnable=True,
                 df_learnable=True):
        super().__init__()
        self.n_dims = len(loc)
        self.loc = torch.tensor(loc)
        if loc_learnable:
            self.loc = Parameter(self.loc)

        self._scale = self.softplus_inverse(torch.tensor(scale))
        if scale_learnable:
            self._scale = Parameter(self._scale)

        self._df = self.softplus_inverse(torch.tensor(df))
        if df_learnable:
            self._df = Parameter(self._df)

    def sample(self, batch_size, return_latents=False):
        weight_model = Gamma(self.df / 2, self.df / 2, learnable=False)
        latent_samples = weight_model.sample(batch_size)
        normal_model = Normal(self.loc.expand(batch_size), self.scale / latent_samples,
                              learnable=False, diag=True)
        if return_latents:
            return normal_model.sample(1).squeeze().unsqueeze(1), latent_samples
        else:
            return normal_model.sample(1).squeeze().unsqueeze(1)

    def log_prob(self, samples, latents=None):
        if latents is None:
            raise NotImplementedError("InfiniteMixtureModel log_prob not implemented")
        weight_model = Gamma(self.df / 2, self.df / 2, learnable=False)
        normal_model = Normal(self.loc.expand(latents.size(0)), self.scale / latents,
                              learnable=False, diag=True)
        return normal_model.log_prob(samples) + weight_model.log_prob(latents)

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def df(self):
        return softplus(self._df)

    @property
    def has_latents(self):
        return True

    def get_parameters(self):
        if self.n_dims == 1:
            return {
                "loc": self.loc.item(),
                "scale": self.scale.item(),
                "df": self.df.item(),
            }
        return {
            "loc": self.loc.detach().numpy(),
            "scale": self.scale.detach().numpy(),
            "df": self.df.detach().numpy(),
        }




# EOF
