import torch
from torch import nn
import numpy as np
import math
from dpm.newton import gradient
from .normal import Normal
from .distribution import Distribution

# Adds langevin dynamics to the model
class Langevin(Distribution):

    def __init__(self, x_t, model, tau=1.):
        super().__init__()
        self.model = model
        self.n_dims = self.model.n_dims
        self.tau = tau
        self.x_t = x_t.float()
        self.noise = Normal(torch.zeros(self.n_dims),
                     (2. * self.tau) * torch.ones(self.n_dims))

    def log_prob(self, value):
        samples = self.x_t.expand_as(value)
        samples.requires_grad = True
        log_probs = self.model.log_prob(samples)
        noise_sample = samples - value + self.tau * gradient(log_probs, samples).squeeze(0)
        tmp = self.noise.log_prob(noise_sample).detach()
        return tmp

    def sample(self, batch_size):
        samples = self.x_t.expand(batch_size, self.n_dims).detach()
        samples.requires_grad = True
        log_probs = self.model.log_prob(samples)
        eta = self.noise.sample(batch_size)
        samples = samples + self.tau * gradient(log_probs, samples).squeeze(0) + eta
        return samples.detach()

    def get_parameters(self):
        return {'distribution':self.model.get_parameters(),
                'tau': self.tau}
