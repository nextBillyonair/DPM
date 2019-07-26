import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from .normal import Normal
from .conditional_model import ConditionalModel
from .dirac_delta import DiracDelta

class Generator(Distribution):

    def __init__(self, latent_distribution=None, input_dim=8,
                 hidden_sizes=[24, 24], activation="LeakyReLU",
                 output_shapes=[1], output_activations=[None]):
        super().__init__()
        self.latent_distribution = latent_distribution
        if latent_distribution is None:
            self.latent_distribution = Normal(torch.zeros(8), torch.eye(8), learnable=False)
        self.conditional_model = ConditionalModel(input_dim, hidden_sizes, activation,
                                                  output_shapes, output_activations, DiracDelta)
        self.n_dims = output_shapes[0]

    def log_prob(self, value):
        raise NotImplementedError("Generator log_prob not implemented")

    def sample(self, batch_size):
        latent_samples = self.latent_distribution.sample(batch_size)
        return self.conditional_model.sample(latent_samples)

    def get_parameters(self):
        return {'latent':self.latent_distribution.get_parameters()}
