from dpm.distributions import Data
from dpm.train import train
from dpm.divergences import cross_entropy

def fit(x, y, model, **kwargs):
    data = Data(x, y)
    stats = train(data, model, cross_entropy, **kwargs)
    return stats

def predict(x, model, compute_logprob=False):
    return model.sample(x, compute_logprob)
