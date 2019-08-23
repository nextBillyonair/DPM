import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
import dpm.utils as utils

class FisherSnedecor(Distribution):

    def __init__(self, df_1=1., df_2=1., learnable=True):
        super().__init__()
        if not isinstance(df_1, torch.Tensor):
            df_1 = torch.tensor(df_1).view(-1)
        self.n_dims = len(df_1)
        if not isinstance(df_2, torch.Tensor):
            df_2 = torch.tensor(df_2).view(-1)
        self._df_1 = utils.softplus_inverse(df_1.float())
        self._df_2 = utils.softplus_inverse(df_2.float())
        if learnable:
            self._df_1 = Parameter(self._df_1)
            self._df_2 = Parameter(self._df_2)

    def log_prob(self, value):
        return dists.FisherSnedecor(self.df_1, self.df_2).log_prob(value).sum(-1)

    def sample(self, batch_size):
        return dists.FisherSnedecor(self.df_1, self.df_2).rsample((batch_size,))

    @property
    def df_1(self):
        return softplus(self._df_1)

    @property
    def df_2(self):
        return softplus(self._df_2)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'df_1':self.df_1.item(),'df_2':self.df_2.item()}
        return {'df_1':self.df_1.detach().numpy(),
                'df_2':self.df_2.detach().numpy()}
