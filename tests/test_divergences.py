import torch
from dpm.divergences import (
    cross_entropy, forward_kl, reverse_kl,
    js_divergence, _forward_kl
)
from dpm.distributions import (
    Normal, Exponential, GumbelSoftmax, Cauchy,
    Beta, LogNormal, Gamma, RelaxedBernoulli, Uniform, StudentT,
    Dirichlet, FisherSnedecor, HalfCauchy, HalfNormal, Laplace,
    Logistic, ChiSquare
)
from dpm.mixture_models import (
    MixtureModel, GumbelMixtureModel
)
import pytest


test_dists = [
    (Normal(0., 1.), Normal(0., 1.)),
    (Exponential(0.5), Exponential(0.5)),
    (Cauchy(0., 1.), Cauchy(0., 1.)),
    (Beta(0.5, 1.), Beta(0.5, 1.)),
    (LogNormal(0., 1.), LogNormal(0., 1.)),
    (Gamma(0.5, 1.), Gamma(0.5, 1.)),
    (Uniform(-1.0, 3.0), Uniform(-1.0, 3.0)),
    (StudentT(30.0, 1.0, 3.0), StudentT(30.0, 1.0, 3.0)),
    (FisherSnedecor(10.0, 10.0), FisherSnedecor(10.0, 10.0)),
    (HalfCauchy(1.0), HalfCauchy(1.0)),
    (HalfNormal(1.0), HalfNormal(1.0)),
    (Laplace(0., 1.), Laplace(0., 1.)),
    (MixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75]),
     MixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75])),
    (GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75]),
     GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75])),
    (GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75], hard=False),
     GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75], hard=False)),
    (Logistic(0., 1.), Logistic(0., 1.)),
    (ChiSquare(4.), ChiSquare(4.)),

    (Normal([0., 0.], [1., 1.], diag=True), Normal([0., 0.], [1., 1.], diag=True)),
    (Exponential([0.5, 0.5]), Exponential([0.5, 0.5])),
    (Cauchy([0., 0.], [1., 1.]), Cauchy([0., 0.], [1., 1.])),
    (Beta([0.5, 0.5], [1., 1.]), Beta([0.5, 0.5], [1., 1.])),
    (LogNormal([0., 0.], [1., 1.]), LogNormal([0., 0.], [1., 1.])),
    (Gamma([0.5, 0.5], [1., 1.]), Gamma([0.5, 0.5], [1., 1.])),
    (Uniform([-1.0, -1.0], [3.0, 3.0]), Uniform([-1.0, -1.0], [3.0, 3.0])),
    (StudentT([30.0, 30.], [1.0, 2.0], [3.0, 3.]), StudentT([30.0, 30.], [1.0, 2.0], [3.0, 3.])),
    (FisherSnedecor([10.0, 10.0],[10.0, 10.0]), FisherSnedecor([10.0, 10.0],[10.0, 10.0])),
    (HalfCauchy([1.0, 1.]), HalfCauchy([1.0, 1.])),
    (HalfNormal([1.0, 1.]), HalfNormal([1.0, 1.])),
    (Laplace([0., 0.], [1., 1.]), Laplace([0., 0.], [1., 1.])),
    (MixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75]),
     MixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75])),
    (GumbelMixtureModel([Normal([0., 0.], [1., 1.], diag=True), Normal([1., 1.], [3., 3.], diag=True)], [0.25, 0.75]),
     GumbelMixtureModel([Normal([0., 0.], [1., 1.], diag=True), Normal([1., 1.], [3., 3.], diag=True)], [0.25, 0.75])),
    (GumbelMixtureModel([Normal([0., 0.], [1., 1.], diag=True), Normal([1., 1.], [3., 3.], diag=True)], [0.25, 0.75], hard=False),
     GumbelMixtureModel([Normal([0., 0.], [1., 1.], diag=True), Normal([1., 1.], [3., 3.], diag=True)], [0.25, 0.75], hard=False)),
    (ChiSquare([4., 4.]), ChiSquare([4., 4.])),
    (Logistic([0., 0.], [1., 1.]), Logistic([0., 0.], [1., 1.])),
]

@pytest.mark.parametrize("p_model,q_model", test_dists)
def test_divergences(p_model, q_model):
    cross_entropy(p_model, q_model, batch_size=1)
    cross_entropy(p_model, q_model, batch_size=64)

    forward_kl(p_model, q_model, batch_size=1)
    forward_kl(p_model, q_model, batch_size=64)

    reverse_kl(p_model, q_model, batch_size=1)
    reverse_kl(p_model, q_model, batch_size=64)

    # Used in JS
    _forward_kl(p_model, q_model, batch_size=1)
    _forward_kl(p_model, q_model, batch_size=64)

    js_divergence(p_model, q_model, batch_size=1)
    js_divergence(p_model, q_model, batch_size=64)
