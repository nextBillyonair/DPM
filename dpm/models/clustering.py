from abc import abstractmethod, ABC
import torch
from dpm.distributions import (
    Distribution, Normal, Data
)
from dpm.mixture_models import MixtureModel
from dpm.train import train
from dpm.divergences import cross_entropy


class GaussianMixture(Distribution):

    def __init__(self, n_components=2, n_dims=1):
        super().__init__()
        self.n_components = n_components
        self.n_dims = n_dims
        self.model = MixtureModel([Normal(torch.randn(n_dims), torch.ones(n_dims))
                                   for _ in range(n_components)],
                                  [1.0 / n_components for _ in range(n_components)])


    def log_prob(self, value):
        return self.model.log_prob(value)

    def sample(self, batch_size):
        return self.model.sample(batch_size)

    def fit(self, x, **kwargs):
        data = Data(x)
        stats = train(data, self.model, cross_entropy, **kwargs)
        return stats

    def predict(self, x):
        log_probs = torch.stack([sub_model.log_prob(x)
                                 for sub_model in self.model.models])
        _, labels = log_probs.max(dim=0)
        return labels
