import dpm.monte_carlo as monte_carlo
from dpm.distributions import (
    Normal, Beta, Exponential, Gamma
)
import pytest

test_means = [
    (Normal(0., 1.), 0),
    (Normal(-7., 1.), -7),
    (Exponential(4.), 0.25),
    (Beta(0.5, 0.5), 0.5),
    (Gamma(0.5, 0.5), 1.),
]

@pytest.mark.parametrize("dist,value", test_means)
def test_monte_carlo_expectation(dist, value):
    assert monte_carlo.expectation(dist, 15000) - value < 1e-1
    assert dist.expectation(15000) - value < 1e-1


test_var = [
    (Normal(0., 1.), 1.),
    (Normal(-7., 1.), 1.),
    (Exponential(4.), 0.0625),
    (Beta(0.5, 0.5), 0.125),
    (Gamma(0.5, 0.5), 2.),
]

@pytest.mark.parametrize("dist,value", test_var)
def test_monte_carlo_variance(dist, value):
    assert monte_carlo.variance(dist, 15000) - value < 1e-1
    assert dist.variance(15000) - value < 1e-1


test_median = [
    (Normal(0., 1.), 0.),
    (Normal(-7., 1.), -7.),
    (Exponential(4.), 0.1732867951),
    (Beta(0.5, 0.5), 0.5),
]

@pytest.mark.parametrize("dist,value", test_median)
def test_monte_carlo_median(dist, value):
    assert monte_carlo.median(dist, 15000) - value < 1e-1
    assert dist.median(15000) - value < 1e-1



def test_monte_carlo_cdf():
    dist = Normal(0., 1.)
    tests = [(0 ,0.5), (1., 0.8413),
             (1.96, 0.975), (-1, 0.1587),
             (-1.96, 0.025)]
    for (c, p) in tests:
        assert monte_carlo.cdf(dist, c, 10000) - p < 1e-2
        assert dist.cdf(c, 10000) - p < 1e-2





# EOF
