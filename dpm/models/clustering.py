from abc import abstractmethod, ABC
import torch
from dpm.distributions import (
    Distribution, Normal, Data,
    GumbelSoftmax, ConditionalModel,
    Categorical
)
from dpm.distributions import MixtureModel
from dpm.train import train
from dpm.criterion import cross_entropy, ELBO
from torch.nn import Softmax, ModuleList
from functools import partial
import numpy as np


class GaussianMixtureModel(Distribution):

    def __init__(self, n_components=2, n_dims=1):
        super().__init__()
        self.n_components = n_components
        self.n_dims = n_dims
        self.model = MixtureModel([Normal(torch.randn(n_dims), torch.eye(n_dims))
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


class VariationalCategorical(ConditionalModel):

    has_latents = True

    def __init__(self, conditional_kwargs={}):
        preset_kwargs = {'input_dim':1, 'hidden_sizes':[24, 24], 'activation':'ReLU',
                         'output_shapes':[2], 'output_activations':[Softmax(dim=-1)],
                         'distribution':partial(GumbelSoftmax, temperature=1.0,
                                        hard=True, learnable=False)}

        preset_kwargs.update(conditional_kwargs)
        super().__init__(**preset_kwargs)


class VariationalGaussianMixtureModel(Distribution):

    has_latents = True

    def __init__(self, n_components=2, n_dims=1, variational_kwargs={}, elbo_kwargs={}):
        super().__init__()
        self.n_components = n_components
        self.n_dims = n_dims
        self.normals = ModuleList([Normal(torch.randn(n_dims), torch.eye(n_dims))
                                   for _ in range(n_components)])
        variational_kwargs.update({'input_dim':n_dims,
                                   'output_shapes':[n_components]})
        self.variational_kwargs = variational_kwargs
        self.elbo_kwargs = elbo_kwargs
        self.categorical = VariationalCategorical(variational_kwargs)
        self.criterion = ELBO(self.categorical, **elbo_kwargs)
        self.prior = Categorical([1.0 / n_components for _ in range(n_components)],
                                 learnable=False)


    def log_prob(self, X, Z=None, n_iter=10):
        if Z is None:

            # Z = self.categorical.sample(X.expand(n_iter, *X.shape))
            # print(Z.shape)
            # raise ValueError()
            Z = self.categorical.sample(X)
            for _ in range(n_iter - 1):
                Z = Z + self.categorical.sample(X)
            Z = Z / n_iter
        latent_probs = self.prior.log_prob(Z)
        log_probs = torch.stack([sub_model.log_prob(X)
                                for sub_model in self.normals], dim=1)
        return (log_probs * Z).sum(dim=-1) + latent_probs


    def sample(self, batch_size):
        indices = self.prior.sample(batch_size).view(-1).long()
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.normals])
        return samples[indices, np.arange(batch_size)]


    def fit(self, x, **kwargs):
        data = Data(x)
        return train(data, self, self.criterion, **kwargs)

    def predict(self, x):
        log_probs = torch.stack([sub_model.log_prob(x)
                                 for sub_model in self.normals])
        _, labels = log_probs.max(dim=0)
        return labels

    def parameters(self):
        for name, param in self.named_parameters(recurse=True):
            if 'categorical' in name:
                continue
            yield param











# EOF
