from abc import abstractmethod, ABC
import numpy as np
from torch.nn import Module

# Base Distribution for all Distributions to inherit from
class Distribution(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_prob(self, value):
        raise NotImplementedError("log_prob method is not implemented")

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError("sample method is not implemented")

    def entropy(self):
        raise NotImplementedError('Entropy not implemented, use monte carlo approximation')

    def perplexity(self):
        return self.entropy().exp()

    def cross_entropy(self, model):
        raise NotImplementedError('Cross Entropy not implemented, use divergence method')

    def cdf(self, c):
        raise NotImplementedError('CDF not implemented, use monte carlo approximation')

    def icdf(self, c):
        raise NotImplementedError('ICDF not implemented')

    @property
    def expectation(self):
        raise NotImplementedError('Expectation not implemented, use monte carlo approximation')

    @property
    def variance(self):
        raise NotImplementedError('Variance not implemented, use monte carlo approximation')

    @property
    def median(self):
        raise NotImplementedError('Median not implemented, use monte carlo approximation')
