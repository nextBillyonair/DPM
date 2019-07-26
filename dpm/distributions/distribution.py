from abc import abstractmethod, ABC
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

    def entropy(self, batch_size=10000):
        raise NotImplementedError('Entropy not implemented, use monte carlo approximation')

    def perplexity(self, batch_size=10000):
        return self.entropy(batch_size).exp()

    def cross_entropy(self, model, batch_size=10000):
        return -model.log_prob(self.sample(batch_size)).mean()

    def softplus_inverse(self, value, threshold=20):
        inv = (value.exp() - 1.0).log()
        inv[value > threshold] = value[value > threshold]
        return inv

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
