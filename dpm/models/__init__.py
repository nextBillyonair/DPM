
from .gans import (
    GenerativeAdversarialNetwork,
    GAN, MMGAN, WGAN, LSGAN
)
from .classification import (
    LogisticRegression, BayesianLogisticRegression,
    SoftmaxRegression
)
from .regression import (
    LinearRegression, L1Regression,
    RidgeRegression, LassoRegression
)
from .model import (
    Model, NeuralModel, LinearModel,
    fit, predict, parameterize
)
