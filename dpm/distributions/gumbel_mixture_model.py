import torch
from torch.nn import ModuleList
from .distribution import Distribution
from .gumbel_softmax import GumbelSoftmax


# Differentiable, Learnable Mixture Weights
class GumbelMixtureModel(Distribution):
    def __init__(self, models, probs, temperature=1.0, hard=True):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.categorical = GumbelSoftmax(probs, temperature, hard)
        self.models = ModuleList(models)

    def log_prob(self, value):
        log_probs = torch.stack([sub_model.log_prob(value)
                                 for sub_model in self.models], dim=-1)
        cat_log_probs = self.categorical.probs.log()
        return torch.logsumexp(log_probs + cat_log_probs, dim=-1)

    def sample(self, batch_size):
        one_hot = self.categorical.sample(batch_size)
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models], dim=1)
        return (samples * one_hot.unsqueeze(-1)).squeeze(-1).sum(-1).view(batch_size, samples.size(-1))

    def cdf(self, value):
        cdfs = torch.stack([sub_model.cdf(value)
                                 for sub_model in self.models], dim=1)
        cat_cdfs = self.categorical.probs
        return torch.sum(cdfs * cat_cdfs, dim=-1)

    def get_parameters(self):
        return {'probs': self.categorical.probs.detach().numpy(),
                'models': [model.get_parameters() for model in self.models]}
