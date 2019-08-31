import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class StudentT(Distribution):

    def __init__(self, df=1., loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df).view(-1)
        self.loc = loc.float()
        self._scale = utils.softplus_inverse(scale.float())
        self._df = utils.softplus_inverse(df.float())
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)
            self._df = Parameter(self._df)

    def log_prob(self, value):
        model = dists.StudentT(self.df, self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = dists.StudentT(self.df, self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self):
        return dists.StudentT(self.df, self.loc, self.scale).entropy()

    @property
    def expectation(self):
        return dists.StudentT(self.df, self.loc, self.scale).mean

    @property
    def variance(self):
        return dists.StudentT(self.df, self.loc, self.scale).variance

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
