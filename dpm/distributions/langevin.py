import torch
from torch import nn
import numpy as np
import math
from dpm.newton import gradient
from .normal import Normal
from .distribution import Distribution

# Uses langevin dynamics to samples
class Langevin(Distribution):

    def __init__(self, distribution=Normal, tau=1., *args, **kwargs):
        super().__init__()
        self.n_dims = distribution.n_dims
        self.distribution = distribution(*args, **kwargs)
        self.tau = tau
        self.noise = Normal(torch.zeros(self.n_dims), torch.ones(self.n_dims))

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def sample(self, batch_size):
        samples = self.distribution.sample(batch_size)
        log_probs = self.distribution.log_prob(samples)
        eta = (2. * self.tau).sqrt() * self.noise.sample(batch_size)
        samples = samples + self.tau * gradient(log_probs) + eta
        return samples

    def get_parameters(self):
        return {'distribution':self.distribution.get_parameters(),
                'tau': self.tau}
