import torch
from dpm.distributions import Data
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
