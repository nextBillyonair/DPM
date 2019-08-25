from abc import abstractmethod, ABC
import torch
from dpm.distributions import Distribution, Data
from dpm.train import train
from dpm.divergences import cross_entropy

def fit(x, y, model, **kwargs):
    data = Data(x, y)
    stats = train(data, model, cross_entropy, **kwargs)
    return stats

def predict(x, model, compute_logprob=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return model.sample(x.float(), compute_logprob)


class Model(Distribution):

    def __init__(self):
        super().__init__()

    def fit(self, x, y, **kwargs):
        return fit(x, y, self, **kwargs)

    def predict(self, x):
        return predict(x, self)
