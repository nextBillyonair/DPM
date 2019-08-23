from abc import abstractmethod, ABC
import torch
from torch import nn
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
from dpm.distributions import Distribution
from .fit import fit as model_fit
from .fit import predict as model_predict

class Model(Distribution):

    def __init__(self):
        super().__init__()

    def fit(self, x, y, **kwargs):
        return model_fit(x, y, self, **kwargs)

    def predict(x):
        return model_predict(x, self)
