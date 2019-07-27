import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from .gamma import Gamma
import dpm.utils as utils

class ChiSquare(Distribution):

    def __init__(self, df=1., learnable=True):
        super().__init__()
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df).view(-1)
        self._df = utils.softplus_inverse(df)
        self.n_dims = len(df)
        if learnable:
            self._df = Parameter(self._df)

    def log_prob(self, value):
        alpha = 0.5 * self.df
        beta = torch.zeros_like(alpha).fill_(0.5)
        model = Gamma(alpha, beta, learnable=False)
        return model.log_prob(value)

    def sample(self, batch_size):
        alpha = 0.5 * self.df
        beta = torch.zeros_like(alpha).fill_(0.5)
        model = Gamma(alpha, beta, learnable=False)
        return model.sample(batch_size)

    @property
    def df(self):
        return softplus(self._df)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'df':self.df.item()}
        return {'df':self.df.detach().numpy()}
