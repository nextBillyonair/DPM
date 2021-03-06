import torch
from torch import nn
from torch import distributions as dists
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution
from .normal import Normal
from functools import partial

# For ELBO!
# AKA Conditional Model
class ConditionalModel(Distribution):

    has_latents = True

    def __init__(self, input_dim=1, hidden_sizes=[24, 24], activation='ReLU',
                 output_shapes=[1, 1], output_activations=[None, 'Softplus'],
                 distribution=partial(Normal, learnable=False)):
        super().__init__()
        if not isinstance(output_shapes, list):
            output_shapes = [output_shapes]
        if not isinstance(output_activations, list):
            output_activations = [output_activations]

        self.input_dim = input_dim
        self.output_dim = output_shapes[0]
        prev_size = input_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(getattr(nn, activation)() if isinstance(activation, str) else activation)
            layers.append(nn.BatchNorm1d(h))
            prev_size = h

        self.model = nn.Sequential(*layers)
        self.output_layers = nn.ModuleList()
        for output_shape, output_activation in zip(output_shapes, output_activations):
            layers = [nn.Linear(prev_size, output_shape)]
            if output_activation is not None:
                layers.append(getattr(nn, output_activation)() if isinstance(output_activation, str) else output_activation)
            self.output_layers.append(nn.Sequential(*layers))

        self.distribution = distribution

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))
        # print(x.shape)
        h = self.model(x)
        h = h.reshape(*batch_shape, -1)
        # print(h.shape)
        return [output_layer(h) for output_layer in self.output_layers]

    def _create_dist(self, x):
        dist_params = self(x)
        # print(len(dist_params), dist_params[0].shape)
        return self.distribution(*dist_params, learnable=False)

    def sample(self, x, compute_logprob=False):
        # sample from q(z|x)
        dist = self._create_dist(x)
        z = dist.sample(1).squeeze(0)
        if compute_logprob:
            return z, dist.log_prob(z)
        return z

    def log_prob(self, z, x):
        # log_prob of q(z|x)
        return self._create_dist(x).log_prob(z)
