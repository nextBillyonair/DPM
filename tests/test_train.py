import torch
from dpm.divergences import (
    cross_entropy, forward_kl, reverse_kl,
    js_divergence, _forward_kl
)
from dpm.distributions import (
    Normal, Exponential, GumbelSoftmax, Cauchy,
    Beta, LogNormal, Gamma, RelaxedBernoulli, Uniform, StudentT,
    Dirichlet, FisherSnedecor, HalfCauchy, HalfNormal, Laplace,
)
from dpm.mixture_models import (
    MixtureModel, GumbelMixtureModel
)
from dpm.train import train

import pytest


test_dists = [
    (Normal(0., 1.), Normal(0., 1.)),
    (Exponential(0.5), Exponential(0.5)),
    (Cauchy(0., 1.), Cauchy(0., 1.)),
    (Beta(0.5, 1.), Beta(0.5, 1.)),
    (LogNormal(0., 1.), LogNormal(0., 1.)),
    (Gamma(0.5, 1.), Gamma(0.5, 1.)),
    (RelaxedBernoulli([0.5]), RelaxedBernoulli([0.5])),
    (Uniform(-1.0, 3.0), Uniform(-1.0, 3.0)),
    (StudentT(30.0, 1.0, 3.0), StudentT(30.0, 1.0, 3.0)),
    (Dirichlet([0.5, 0.5]), Dirichlet([0.5, 0.5])),
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
]

@pytest.mark.parametrize("p_model,q_model", test_dists)
def test_train(p_model, q_model):

    train(p_model, q_model, cross_entropy, epochs=3)
    train(p_model, q_model, forward_kl, epochs=3)
    train(p_model, q_model, reverse_kl, epochs=3)
    train(p_model, q_model, _forward_kl, epochs=3)
    train(p_model, q_model, js_divergence, epochs=3)
