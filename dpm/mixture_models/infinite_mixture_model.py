import torch
from torch.distributions import Categorical
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus
import numpy as np

from dpm.distributions import Distribution, Gamma, Normal
import dpm.utils as utils

# Infinite Mixture Version of Gaussian
class InfiniteMixtureModel(Distribution):
    def __init__(self, df, loc, scale,
                 loc_learnable=True,
                 scale_learnable=True,
                 df_learnable=True):
        super().__init__()
        self.loc = torch.tensor(loc).view(-1)
        self.n_dims = len(self.loc)
        if loc_learnable:
            self.loc = Parameter(self.loc)
        self._scale = utils.softplus_inverse(torch.tensor(scale).view(-1))
        if scale_learnable:
            self._scale = Parameter(self._scale)
        self._df = utils.softplus_inverse(torch.tensor(df).view(-1))
        if df_learnable:
            self._df = Parameter(self._df)

    def sample(self, batch_size, return_latents=False):
        weight_model = Gamma(self.df / 2, self.df / 2, learnable=False)
        latent_samples = weight_model.sample(batch_size)
        normal_model = Normal(self.loc.expand(batch_size), (self.scale / latent_samples).squeeze(1),
                              learnable=False, diag=True)
        if return_latents:
            return normal_model.sample(1).squeeze(0).unsqueeze(1), latent_samples
        else:
            return normal_model.sample(1).squeeze(0).unsqueeze(1)

    def log_prob(self, samples, latents=None):
        if latents is None:
            raise NotImplementedError("InfiniteMixtureModel log_prob not implemented")
        weight_model = Gamma(self.df / 2, self.df / 2, learnable=False)
        normal_model = Normal(self.loc.expand(latents.size(0)), (self.scale / latents).squeeze(1),
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
