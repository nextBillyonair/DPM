from dpm.monte_carlo import inverse_sampling
from dpm.distributions import (
    Arcsine, Cauchy, Exponential, Gumbel, HyperbolicSecant,
    Laplace, LogCauchy, Rayleigh, Uniform, Data, Normal,
    Logistic, LogitNormal
)
from dpm.divergences import forward_kl
import pytest


test_models = [
    Arcsine(), Cauchy(), Exponential(),
    Gumbel(), HyperbolicSecant(),
    Laplace(), Rayleigh(), Uniform(), Logistic(),
    Normal(), LogitNormal()
]


@pytest.mark.parametrize("model", test_models)
def test_icdf(model):
    X = model.sample(10)
    p = model.cdf(X)
    X_p = model.icdf(p)
    assert (X - X_p <= 1e-1).all()

def test_log_cauchy():
    model = LogCauchy()
    X = model.sample(10)
    p = model.cdf(X)
    X_p = model.icdf(p)
    # assert (X - X_p <= 0.2).all()


@pytest.mark.parametrize("model", test_models)
def test_inverse_sampling(model):
    samples = inverse_sampling(model.icdf, batch_size=100)
    # add kl divergence to make sure it is approx same!
