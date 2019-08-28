import torch
from functools import partial
from dpm.distributions import (
    Normal, Laplace
)
from .model import LinearModel


################################################################################
# LINEAR REGRESSION
################################################################################

class LinearRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)))


class L1Regression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape,
            distribution=partial(Laplace, scale=torch.ones(output_shape)))

# Bayesian Linear Regression
class RidgeRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)),
            prior=Normal(0., tau))


class LassoRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim, output_shape,
            distribution=partial(Normal, scale=torch.ones(1, output_shape)),
            prior=Laplace(0., tau))



# EOF