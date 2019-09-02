import pytest
from dpm.distributions import *
import dpm.utils as utils


def test_arcsine():
    model = Arcsine()
    assert model.expectation == 0.5
    assert model.median == 0.5
    assert model.variance == 0.125
    assert model.skewness == 0.
    assert model.kurtosis == -1.5

    model = Arcsine(-1, 1)
    assert model.expectation == 0.
    assert model.median == 0.
    assert model.variance == 0.5
    assert model.skewness == 0.
    assert model.kurtosis == -1.5

def test_bernoulli():
    model = Bernoulli(probs=[0.3])
    assert model.logits.item() + 1.2039728043 < 1e-2
    assert model.expectation.item() - 0.3 < 1e-2
    assert model.variance.item() - 0.21 < 1e-2
    assert model.skewness.item() - 1.9047619048 < 1e-2
    assert model.kurtosis.item() + -1.2380952381 < 1e-2

def test_beta():
    model = Beta()
    assert model.expectation == 0.5
    assert model.variance == 0.125
    m = Beta(0.5, 0.5).mode.item()
    assert m == 0. or 1.
    assert Beta(4.5, 3.5).mode.item() - 0.5833333333 < 1e-2
    assert Beta(1.5, 0.5).mode.item() == 1.
    assert Beta(0.5, 1.5).mode.item() == 0.
    # assert Beta(1.00000, 1.00000).mode.item() > 0. and Beta(1.00000, 1.00000).mode.item() < 1.

def test_cauchy():
    model = Cauchy(loc=1.)
    assert model.median == 1.
    assert model.mode == 1.

def test_exponential():
    model = Exponential()
    assert model.expectation - 1. < 1e-2
    assert model.mode - 0. < 1e-2
    assert model.variance - 1. < 1e-2
    assert model.median - 0.6931471806 < 1e-2
    assert model.skewness - 2. < 1e-2
    assert model.kurtosis - 6. < 1e-2

    model = Exponential(0.5)
    assert model.expectation - 2. < 1e-2
    assert model.mode - 0. < 1e-2
    assert model.variance - 4. < 1e-2
    assert model.median - 1.3862943611 < 1e-2
    assert model.skewness - 2. < 1e-2
    assert model.kurtosis - 6. < 1e-2

def test_gamma():
    model = Gamma()
    assert model.expectation - 1. < 1e-2
    assert model.variance - 1. < 1e-2

    model = Gamma(0.5, 0.75)
    assert model.expectation - 0.6666666667 < 1e-2
    assert model.variance - 0.8888888889 < 1e-2

def test_gumbel():
    model = Gumbel(loc=1., scale=2.)
    assert model.expectation - (1 + 2 * utils.euler_mascheroni) < 1e-2
    assert model.mode == 1.
    assert model.median - 1.7330258412 < 1e-2
    assert model.variance - 6.5797362674 < 1e-2
    assert model.skewness - 1.14 < 1e-2
    assert model.kurtosis - 2.4 < 1e-2

def test_hyperbolicsecant():
    model = HyperbolicSecant()
    assert model.expectation == 0.
    assert model.variance == 1.
    assert model.median == 0.

def test_laplace():
    model = Laplace(loc=1., scale=2.)
    assert model.expectation - 1. < 1e-2
    assert model.variance - 8. < 1e-2
    assert model.stddev - 2.8284271247 < 1e-2
    assert model.median - 1. < 1e-2
    assert model.mode - 1. < 1e-2
    assert model.skewness < 1e-2
    assert model.kurtosis - 3. < 1e-2
    assert model.entropy() - 2.3862943611 < 1e-2

def test_log_cauchy():
    model = LogCauchy(loc=2.)
    assert model.median - 7.3890560989 < 1e-2

def test_log_normal():
    model = LogNormal()
    assert model.expectation - 1.6487212707 < 1e-2
    assert model.variance - 4.6707742705 < 1e-2
    assert model.mode - utils.e < 1e-2
    assert model.median - utils.e < 1e-2

def test_logistic():
    model = Logistic(loc=1., scale=2.)
    assert model.expectation == 1.
    assert model.mode == 1.
    assert model.variance - 13.1594725348 < 1e-2
    assert model.median == 1.
    assert model.skewness == 0.
    assert model.kurtosis == 1.2

def test_rayleigh():
    model = Rayleigh(3.)
    assert model.expectation - 3.7599424119 < 1e-2
    assert model.mode - 3. < 1e-2
    assert model.median - 3.5322300675 < 1e-2
    assert model.variance - 3.8628330588 < 1e-2
    assert model.skewness - 1.1186145158 < 1e-2
    assert model.kurtosis - 0.2450893007 < 1e-2


def test_studentt():
    model = StudentT()
    model.expectation
    model.variance

def test_uniform():
    model = Uniform()
    assert model.expectation - 0.5 < 1e-2
    assert model.variance - 1/12. < 1e-2
    assert model.median - 0.5 < 1e-2
    assert model.skewness == 0.
    assert model.kurtosis + 1.2 < 1e-2


















# EOF
