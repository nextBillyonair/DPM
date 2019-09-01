import dpm.monte_carlo as monte_carlo
from dpm.distributions import (
    Normal, Beta, Exponential, Gamma, Uniform, Laplace
)
import pytest
from scipy.stats import ks_2samp

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


test_var = [
    (Normal(0., 1.), 1.),
    (Normal(-7., 1.), 1.),
    (Exponential(4.), 0.0625),
    (Beta(0.5, 0.5), 0.125),
    (Gamma(0.5, 0.5), 2.),
]

@pytest.mark.parametrize("dist,value", test_var)
def test_monte_carlo_variance(dist, value):
    assert monte_carlo.variance(dist, 15000) - value < 0.2


test_median = [
    (Normal(0., 1.), 0.),
    (Normal(-7., 1.), -7.),
    (Exponential(4.), 0.1732867951),
    (Beta(0.5, 0.5), 0.5),
]

@pytest.mark.parametrize("dist,value", test_median)
def test_monte_carlo_median(dist, value):
    assert monte_carlo.median(dist, 15000) - value < 0.2



def test_monte_carlo_cdf():
    dist = Normal(0., 1.)
    tests = [(0 ,0.5), (1., 0.8413),
             (1.96, 0.975), (-1, 0.1587),
             (-1.96, 0.025)]
    for (c, p) in tests:
        assert monte_carlo.cdf(dist, c, 10000) - p < 1e-2

test_entropy = [
    Normal(0., 1.),
    Normal(-7., 1.),
    Exponential(4.),
    Beta(0.5, 0.5),
    Gamma(0.5, 0.5)
]

@pytest.mark.parametrize("dist", test_entropy)
def test_monte_carlo_entropy(dist):
    assert dist.entropy() - monte_carlo.entropy(dist, 15000) < 0.2


def test_max_min():
    model = Uniform(-1., 5.)
    assert monte_carlo.max(model) - 5. < 1e-1
    assert monte_carlo.min(model) + 1. < 1e-1


def test_rejection_sampling():
    samples = monte_carlo.rejection_sampling(Normal(-7.3, 3.),
                                Normal(-7.3, 3., learnable=False),
                                10,
                                batch_size=100000)

    assert samples.mean() + 7.3 < 0.2
    assert samples.var() - 3. < 0.2


def test_box_muller():
    Z1, Z2 = monte_carlo.box_muller()
    assert Z1.mean() - 0.0 < 0.05
    assert Z2.mean() - 0.0 < 0.05
    assert Z1.var() - 1.0 < 0.1
    assert Z2.var() - 1.0 < 0.1

def test_margaglia_bray():
    Z1, Z2 = monte_carlo.marsaglia_bray()
    assert Z1.mean() - 0.0 < 0.05
    assert Z2.mean() - 0.0 < 0.05
    assert Z1.var() - 1.0 < 0.1
    assert Z2.var() - 1.0 < 0.1


def test_lcg():
    generator = monte_carlo.lcg(modulus=9, a=4, c=1, seed=0)
    values = [1, 5, 3, 4, 8, 6, 7, 2, 0]
    for i in values:
        assert next(generator) == i


def test_rand_generator():
    normal_samples = Uniform().sample(10000).detach().view(-1)
    samples = monte_carlo.rand(batch_size=10000).view(-1)
    stat, p_value = ks_2samp(samples.numpy(), normal_samples.numpy())
    assert stat <= 0.1
    assert p_value >= 0.03


def test_monet_carlo_no_errors():
    model = Laplace()
    mean = model.loc .detach()
    scale = model.scale.detach()
    monte_carlo.expectation(model)
    monte_carlo.variance(model)
    monte_carlo.standard_deviation(model)
    monte_carlo.median(model)
    monte_carlo.skewness(model)
    monte_carlo.kurtosis(model)
    monte_carlo.entropy(model)
    monte_carlo.cdf(model, 0.)




# EOF
