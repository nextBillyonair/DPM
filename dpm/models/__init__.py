
from .gans import (
    GenerativeAdversarialNetwork,
    GAN, MMGAN, WGAN, LSGAN
)
from .classification import (
    LogisticRegression, BayesianLogisticRegression,
    SoftmaxRegression
)
from .clustering import (
    GaussianMixture
)
from .decomposition import (
    pca, PCA,
    EMPPCA,
    #ProbabilisticPCA
)
from .factorization import (
    PMF
)
from .naive_bayes import (
    NaiveBayes,
    GaussianNaiveBayes,
    BernoulliNaiveBayes,
    MultinomialNaiveBayes
)
from .regression import (
    LinearRegression, L1Regression,
    RidgeRegression, LassoRegression
)
from .model import (
    Model, NeuralModel, LinearModel,
    fit, predict, parameterize
)
# from .vae import (
#   Encoder, Decoder, VAE
# )
