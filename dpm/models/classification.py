from torch.nn import Softmax
from dpm.distributions import (
    Bernoulli, Categorical, Normal
)
from dpm.utils import Sigmoid
from .model import LinearModel

################################################################################
# LOGISTIC REGRESSION
################################################################################

class LogisticRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1):
        super().__init__(input_dim, output_shape, output_activation=Sigmoid(),
            distribution=Bernoulli)

# Normal prior on weights
class BayesianLogisticRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=1, tau=1.):
        super().__init__(input_dim, output_shape, output_activation=Sigmoid(),
            distribution=Bernoulli, prior=Normal(0., tau))


################################################################################
# MULTICLASS CLASSIFICATION
################################################################################

# multiclass, max ent
class SoftmaxRegression(LinearModel):

    def __init__(self, input_dim=1, output_shape=2):
        super().__init__(input_dim, output_shape, output_activation=Softmax(dim=1),
            distribution=Categorical)


# EOF
