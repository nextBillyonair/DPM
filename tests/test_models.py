from dpm.models import (
    LinearRegression, L1Regression,
    RidgeRegression, LassoRegression,
    LogisticRegression, BayesianLogisticRegression,
    SoftmaxRegression, PMF,
    GaussianMixture,
    GAN, WGAN, LSGAN, MMGAN
)
from dpm.distributions import Normal, Categorical
from dpm.mixture_models import MixtureModel
from sklearn import datasets
import numpy as np
import torch
import pytest

regression_models = [
    LinearRegression, L1Regression,
    RidgeRegression, LassoRegression
]

def gen():
    N = 1000

    x = np.random.uniform(-10, 10, (N, 1))
    x = np.concatenate((x**3, x**2, x), axis = 1)
    # print(x[:, 1])
    w = np.array([[3.4], [1.6], [-4.5]])
    b = -4.2
    y = x @ w  + b + np.random.normal(0, 1, (N, 1))
    return x, y

@pytest.mark.parametrize("model", regression_models)
def test_regression(model):
    x, y = gen()
    model = model(input_dim=3)
    model.fit(x, y, epochs=150, lr=0.1, batch_size=1024)
    y_pred = model.predict(x)
    assert y_pred.shape == y.shape


logistic_models = [
    LogisticRegression, BayesianLogisticRegression
]
@pytest.mark.parametrize("model", logistic_models)
def test_logistic(model):
    x, y = gen()
    y = torch.sigmoid(torch.tensor(y))
    y = y.round().float()
    model = model(input_dim=3)
    model.fit(x, y, epochs=200)
    assert (y == model.predict(x)).float().mean() >= 0.85


def test_classification():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    model = SoftmaxRegression(input_dim=4, output_shape=3)
    stats = model.fit(X, y, epochs=2000)
    y_pred = model.predict(X)[0]
    assert (y_pred.numpy() == y).mean() >= 0.85

def build_toy_dataset(U, V, N, M, noise_std=0.1):
    R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
    return R

factor_models = [
    PMF(60, 50, 3),PMF(60, 50, 3, tau=10.)
]
@pytest.mark.parametrize("model", factor_models)
def test_factorization(model):
    N, M, D = (60, 50, 3)
    U_true = np.random.randn(D, N)
    V_true = np.random.randn(D, M)
    R_true = np.stack([build_toy_dataset(U_true, V_true, N, M) for _ in range(10)])
    R_true = torch.tensor(R_true).float()

    model.fit(R_true, epochs=400, lr=0.01)

    assert model.sample(3).shape == (3, 60, 50)
    assert model.sample(1).shape == (1, 60, 50)
    assert model.reconstruct().shape == (60, 50)
    assert model.log_prob(R_true).shape == (10, )
    assert model.mse(R_true) < 0.1
    assert model.mae(R_true) < 0.1

    # not pytorch
    assert model.log_prob(R_true.numpy()).shape == (10, )
    assert model.mse(R_true.numpy()) < 0.1
    assert model.mae(R_true.numpy()) < 0.15


def test_gmm_clustering():
    model = MixtureModel([Normal([3.3, 3.3], [2.3, 0.1, 0.1, 7.]),
                      Normal([-5.3, -6.3], [7, 4.2, 3.1, 3])], [0.75, 0.25])

    X = model.sample(100).detach()
    m = GaussianMixture(n_dims=2)
    m.fit(X, epochs=100, track_parameters=False)
    assert m.sample(5).shape == (5, 2)
    assert m.log_prob(m.sample(5)).shape == (5, )
    assert m.predict(X).shape == (100, )

gans = [
    GAN(),
    WGAN(),
    LSGAN(),
    MMGAN(),
    GAN(criterion_args={'grad_penalty':10.}),
    GAN(criterion_args={'use_spectral_norm':True})
]
@pytest.mark.parametrize("model", gans)
def test_gans(model):
    X = MixtureModel([Normal(-4., 2.3, learnable=False), Normal(4., 2.3, learnable=False)], [0.5, 0.5]).sample(10000)
    X = X.numpy()
    stats = model.fit(X, epochs=5, lr=1e-4)
    preds = model.sample(10000)
    model.predict(model.sample(100))
    try:
        model.log_prob(model.sample(100))
    except NotImplementedError:
        pass


# EOF
