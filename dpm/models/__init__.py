
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
    ProbabilisticPCA, PPCA_Variational, PPCA_Variational_V2
)
from .factorization import (
    PMF
)
from .generative_classification import (
    GenerativeClassifier,
    GaussianNaiveBayes,
    BernoulliNaiveBayes,
    MultinomialNaiveBayes,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from .regression import (
    LinearRegression, L1Regression,
    RidgeRegression, LassoRegression,
    PoissonRegression, NegativeBinomialRegression,
    # BinomialRegression
)
from .model import (
    Model, NeuralModel, LinearModel,
    fit, predict, parameterize
)
from .vae import (
    VAE
)
