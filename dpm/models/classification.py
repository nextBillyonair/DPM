from torch.nn import Softmax
from dpm.distributions import (
    Bernoulli, Categorical, Normal
)
from dpm.utils import Sigmoid
from .model import LinearModel
from functools import partial

################################################################################
# LOGISTIC REGRESSION
################################################################################

class LogisticRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim=input_dim, output_shapes=output_shape,
            output_activations=Sigmoid(),
            distribution=partial(Bernoulli, learnable=False))

# Normal prior on weights
class BayesianLogisticRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim=input_dim, output_shapes=output_shape,
            output_activations=Sigmoid(),
            distribution=partial(Bernoulli, learnable=False),
            prior=Normal(0., tau, learnable=False))


################################################################################
# MULTICLASS CLASSIFICATION
################################################################################

# multiclass, max ent
class SoftmaxRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=2):
        super().__init__(input_dim=input_dim, output_shapes=output_shape,
            output_activations=Softmax(dim=-1),
            distribution=partial(Categorical, learnable=False))


# EOF
