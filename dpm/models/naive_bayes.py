import torch
from torch.nn import ModuleList
from dpm.distributions import (
    Bernoulli, Categorical, Normal, Data, Distribution
)
from dpm.train import train
from dpm.divergences import cross_entropy
import numpy as np

# BASE

class NaiveBayes(Distribution):

    def __init__(self, y_dist, x_dist):
        super().__init__()
        self.y_dist = y_dist
        self.x_dist = ModuleList(x_dist)

    def log_prob(self, x, y):
        ids = y.long()
        log_probs = torch.cat([sub_model.log_prob(x).view(-1, 1)
                                 for sub_model in self.x_dist], dim=1)
        y_probs = self.y_dist.log_prob(y).view(-1, 1)
        return (y_probs + log_probs.gather(1, ids.view(-1, 1))).sum(-1)

    def sample(self, batch_size):
        indices = self.y_dist.sample(batch_size).view(-1).long()
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.x_dist])
        # if you want class, return indicies as well
        return samples[indices, np.arange(batch_size)]

    def fit(self, x, y, **kwargs):
        data = Data(x, y)
        stats = train(data, self, cross_entropy, **kwargs)
        return stats

    def predict(self, x):
        log_probs = torch.cat([sub_model.log_prob(x).view(-1, 1)
                                 for sub_model in self.x_dist], dim=1)
        y_probs = self.y_dist.logits.expand_as(log_probs)
        probs = y_probs + log_probs
        return probs.max(dim=1)[1].view(-1, 1)


# Specific Models

class GaussianNaiveBayes(NaiveBayes):

    def __init__(self, n_classes=2, n_features=10):
        super().__init__(Categorical(probs=[1.0/n_classes for _ in range(n_classes)]),
                         [Normal(loc=torch.randn(n_features), scale=torch.ones(n_features)) for _ in range(n_classes)])
        self.n_classes = n_classes
        self.n_features = n_features

class BernoulliNaiveBayes(NaiveBayes):

    def __init__(self, n_features=10):
        super().__init__(Bernoulli(probs=torch.rand(1)),
                         [Bernoulli(probs=torch.rand(n_features)) for _ in range(2)])
        self.n_classes = 2
        self.n_features = n_features

# TODO: Multinomial NB
# y ~ Bernoulli
# Each X feature ~ Multinomial/Categorical

class MultinomialNaiveBayes(NaiveBayes):

    def __init__(self, n_classes=2, n_features=10, n_states=4):
        super().__init__(Categorical(probs=[1.0/n_classes for _ in range(n_classes)]),
                        [Categorical(probs=[[0.5 for _ in range(n_states)]
                                            for _ in range(n_features)])
                                            for _ in range(n_classes)])
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_states = n_states
