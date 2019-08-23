import torch
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math
from .distribution import Distribution


class Data(Distribution):

    def __init__(self, *data, learnable=False):
        super().__init__()
        torch_data = []
        for i, d in enumerate(data):
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d).float()
            torch_data.append(d)

        assert (np.array([d.shape for d in torch_data]) == torch_data[0].shape).all()
        self.n_dims = torch_data[0].size(-1)
        self.n_samples = len(torch_data[0])
        self.n_pairs = len(data)
        self.data = torch_data

    def log_prob(self, value):
        raise NotImplementedError("Data Distribution log_prob not implemented")

    def sample(self, batch_size):
        idx = torch.tensor(np.random.choice(self.data[0].size(0), size=batch_size))
        samples = tuple(d[idx] for d in self.data)
        if self.n_pairs == 1:
            return samples[0]
        return samples

    def get_parameters(self):
        return {'n_dims':self.n_dims, 'n_samples':self.n_samples}
