import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Convolution(Distribution):
    def __init__(self, models, learnable=False):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.models = ModuleList(models)
        self.n_models = len(models)

    def log_prob(self, value):
        raise NotImplementedError("Convolution log_prob not implemented")

    def sample(self, batch_size):
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models])
        return samples.sum(0)

    def get_parameters(self):
        return {'models':[model.get_parameters() for model in self.models]}
