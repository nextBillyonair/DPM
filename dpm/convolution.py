import torch
from torch.nn import ModuleList
from torch.nn.functional import softplus

from dpm.distributions import Distribution

class Convolution(Distribution):
    def __init__(self, models):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.models = ModuleList(models)
        self.n_models = len(models)

    def log_prob(self, value):
        raise NotImplementedError("Convolution log_prob not implemented")

    def sample(self, batch_size):
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models])
        return samples.sum(0)

    def get_parameters(self):
        return {'models':[model.get_parameters() for model in self.models]}
