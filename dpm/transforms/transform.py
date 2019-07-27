from abc import abstractmethod, ABC
from torch.nn import Module


class Transform(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward not implemented")

    @abstractmethod
    def inverse(self, y):
        raise NotImplementedError("Inverse not implemented")

    @abstractmethod
    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError("Log Abs Det Jacobian not implemented")

    def get_parameters(self):
        raise NotImplementedError('Get Parameters not implemented')
