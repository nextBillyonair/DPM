import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Data(Distribution):

    def __init__(self, data, learnable=False):
        super().__init__()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.n_dims = data.size(-1)
        self.n_samples = len(data)
        self.data = data

    def log_prob(self, value):
        raise NotImplementedError("Data Distribution log_prob not implemented")

    def sample(self, batch_size):
        idx = torch.tensor(np.random.choice(self.data.size(0), size=batch_size))
        return self.data[idx]

    def get_parameters(self):
        return {'n_dims':self.n_dims, 'n_samples':self.n_samples}
