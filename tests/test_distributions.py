from dpm.distributions import (
    Arcsine, AsymmetricLaplace, Bernoulli, Beta, Categorical, Cauchy, ChiSquare, Convolution, Data,
    DiracDelta, Dirichlet, Exponential, FisherSnedecor, Gamma, Generator,
    GumbelSoftmax, Gumbel, HalfCauchy, HalfNormal, HyperbolicSecant, Langevin,
    Laplace, LogLaplace, LogCauchy, LogNormal, Logistic, LogitNormal, Normal, Rayleigh,
    RelaxedBernoulli, StudentT, Uniform, Distribution
)
from dpm.mixture_models import (
    MixtureModel, GumbelMixtureModel
)
import torch.distributions as dist
import numpy as np
import torch
import pytest


test_normal_dists = [
    (Normal(0., 1.), 1),
    (Normal([0.], [1.]), 1),
    (Normal([0.], [[1.]]), 1),
    (Normal([0., 0.], [1., 1.]), 2),
    (Normal([0., 0.], [[1., 0.], [0., 1.]]), 2),
    (Normal([0., 0.], [1., 0., 0., 1.]), 2),
]


@pytest.mark.parametrize("dist,n_dims", test_normal_dists)
def test_normal_dist(dist, n_dims):
    assert dist.sample(1).shape == (1, n_dims)
    assert dist.log_prob(dist.sample(1)).shape == (1, )

    samples = dist.sample(64)
    assert samples.shape == (64, n_dims)

    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (64, )

    dist.get_parameters()
    dist.num_parameters
    try:
        dist.entropy()
    except NotImplementedError:
        pass
    try:
        dist.perplexity()
    except NotImplementedError:
        pass


test_normal_dists = [
    (Normal(0., 1.), 1),
    (Normal([0.], [1.]), 1),
    (Normal([0.], [[1.]]), 1),
    (Normal([0., 0.], [1., 1.]), 2),
    (Normal([0., 0.], [[1., 0.], [0., 1.]]), 2),
    (Normal([0., 0.], [1., 0., 0., 1.]), 2),
]


@pytest.mark.parametrize("normal_dist,n_dims", test_normal_dists)
def test_normal_dist_params(normal_dist, n_dims):

    if n_dims == 1:
        assert normal_dist.loc == 0.
        assert (normal_dist.scale - torch.Tensor([1.]) <= 0.01).all()
        params = normal_dist.get_parameters()
        assert params["loc"] == 0.
        assert (params["scale"] - torch.Tensor([1.]) <= 0.01).all()
    elif n_dims == 2:
        params = normal_dist.get_parameters()
        assert (params['loc'] == np.array([0., 0.])).all()
        assert (params['scale'] - np.array([[1., 0.], [0., 1.]]) <= 0.01).all()
        assert (normal_dist.scale - torch.Tensor([[1., 0.], [0., 1.]]) <= 0.01).all()
    else:
        raise ValueError()



test_dists = [
    (Normal(0., 1.), 1),
    (Exponential(0.5), 1),
    (Cauchy(0., 1.), 1),
    (Beta(0.5, 1.), 1),
    (LogNormal(0., 1.), 1),
    (Gamma(0.5, 1.), 1),
    (RelaxedBernoulli([0.5]), 1),
    (Uniform(-1., 3.), 1),
    (StudentT(30., 1., 3.), 1),
    (Dirichlet(0.5), 1),
    (FisherSnedecor(10., 10.), 1),
    (HalfCauchy(1.), 1),
    (HalfNormal(1.), 1),
    (Laplace(0., 1.), 1),
    (MixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75]), 1),
    (GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75]), 1),
    (GumbelMixtureModel([Normal(0., 1.), Normal(1., 3.)], [0.25, 0.75], hard=False), 1),
    (ChiSquare(4.), 1),
    (Logistic(0., 1.), 1),
    (Rayleigh(), 1),
    (LogLaplace(), 1),
    (LogCauchy(), 1),
    (Categorical(), 1),
    (HyperbolicSecant(), 1),
    (Arcsine(), 1),
    (Bernoulli(), 1),
    (Gumbel(), 1),
    (Rayleigh(), 1),
    (Arcsine(), 1),
    (Categorical(), 1),
    (LogitNormal(), 1),
    (AsymmetricLaplace(), 1),
    (AsymmetricLaplace(asymmetry=2.), 1),

    (Normal([0., 0.], [1., 0., 0., 1.0]), 2),
    (Exponential([0.5, 1.0]), 2),
    (Cauchy([0., 0.], [1., 1.]), 2),
    (Beta([0.5, 0.5], [1., 1.]), 2),
    (LogNormal([0., 0.], [1., 1.]), 2),
    (Gamma([0.5, 0.5], [1., 1.]), 2),
    (Gumbel([0., 0.], [1., 1.]), 2),
    (RelaxedBernoulli([0.5, 0.5]), 2),
    (Uniform([-1., -2.], [3., 4.]), 2),
    (StudentT([30., 10.], [1., 2.], [3., 4.]), 2),
    (Dirichlet([0.5, 1.5]), 2),
    (FisherSnedecor([10., 11.], [10., 11.]), 2),
    (HalfCauchy([1., 2.]), 2),
    (HalfNormal([1., 2.]), 2),
    (LogLaplace([0., 0.], [1., 1.]), 2),
    (LogCauchy([0., 0.], [1., 1.]), 2),
    (LogNormal([0., 0.], [1., 1.]), 2),
    (Laplace([0., 0.], [1., 1.]), 2),
    (LogitNormal([0., 0.], [1., 1.]), 2),
    (LogitNormal([0., 0.], [1., 0., 0., 1.0]), 2),
    (MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75]), 2),
    (GumbelMixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75]), 2),
    (GumbelMixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75], hard=False), 2),
    (ChiSquare([4., 9.]), 2),
    (Rayleigh([1., 1.]), 2),
    (Arcsine([0., 0.], [1., 1.]), 2),
    (Bernoulli([0.3, 0.7]), 2),
    (Categorical(probs=[[0.5, 0.5], [0.5, 0.5]]), 2),
]

@pytest.mark.parametrize("dist,n_dims", test_dists)
def test_shapes(dist, n_dims):
    assert dist.sample(1).shape == (1, n_dims)
    assert dist.log_prob(dist.sample(1)).shape == (1, )

    samples = dist.sample(64)
    assert samples.shape == (64, n_dims)

    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (64, )
    dist.num_parameters

    dist.get_parameters()
    try:
        dist.entropy()
    except NotImplementedError:
        pass


gumbels = [
    (GumbelSoftmax([0.5, 0.5]), 2),
    (GumbelSoftmax([0.5, 0.5], hard=False), 2),
    (GumbelSoftmax([0.5, 0.25, 0.25]), 3),
    (GumbelSoftmax([0.5, 0.25, 0.25], hard=False), 3),
]
@pytest.mark.parametrize("dist,n_dims", gumbels)
def test_gumbel_softmax(dist, n_dims):
    assert dist.sample(1).shape == (1, n_dims)
    assert dist.log_prob(dist.sample(1)).shape == (1, )

    samples = dist.sample(64)
    assert samples.shape == (64, n_dims)

    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (64, )

    dist.get_parameters()
    try:
        dist.entropy()
    except NotImplementedError:
        pass


dirac_locs = [
    (1.0, 1),
    ([2.0, 3.0], 2),
    ([1.0, 2.0, 3.0], 3),
]
@pytest.mark.parametrize("loc,n_dims", dirac_locs)
def test_dirac_delta(loc, n_dims):
    dist = DiracDelta(loc)
    assert dist.sample(1).shape == (1, n_dims), "{} {}".format(dist.sample(1).shape, n_dims)

    try:
        dist.log_prob(dist.sample(64))
    except NotImplementedError:
        pass

    samples = dist.sample(64)
    assert samples.shape == (64, n_dims)

    if n_dims == 1:
        assert dist.get_parameters()['loc'] == 1.0
    elif n_dims == 2:
        assert (dist.get_parameters()['loc'] == np.array([2.0, 3.0])).all()
    elif n_dims == 3:
        assert (dist.get_parameters()['loc'] == np.array([1.0, 2.0, 3.0])).all()
    else:
        raise ValueError()
    dist.get_parameters()


def test_dirac_delta_2d():
    dist = DiracDelta([[1,0],[0,1]])
    assert dist.sample(1).shape == (1, 2, 2), "{} {}".format(dist.sample(1).shape, n_dims)
    dist.get_parameters()



dim_list = [1, 2, 3]
@pytest.mark.parametrize("n_dims", dim_list)
def test_data_dist(n_dims):

    data = torch.randn(1000, n_dims)
    dist = Data(data)

    assert dist.sample(1).shape == (1, n_dims)
    assert dist.sample(64).shape == (64, n_dims)

    try:
        dist.log_prob(dist.sample(64))
    except NotImplementedError:
        pass

    assert dist.get_parameters()['n_dims'] == n_dims
    assert dist.get_parameters()['n_samples'] == 1000

    data = np.random.randn(100, n_dims)
    dist = Data(data)

    assert dist.sample(1).shape == (1, n_dims)
    assert dist.sample(64).shape == (64, n_dims)

    assert dist.get_parameters()['n_dims'] == n_dims
    assert dist.get_parameters()['n_samples'] == 100

dists = [
    (Convolution([Normal(15., 1.0), Normal(-10., 1.0)]),1),
    (Convolution([Normal([0., 1.], [1., 1.], ),
                  Normal([-10., 10.], [1.0, 1.0])]), 2)
]
@pytest.mark.parametrize("dist,n_dims", dists)
def test_convolution(dist, n_dims):
    assert dist.sample(1).shape == (1, n_dims)
    assert dist.sample(64).shape == (64, n_dims)

    try:
        dist.log_prob(dist.sample(64))
    except NotImplementedError:
        pass

    dist.get_parameters()

dists_init = [
    Arcsine, Bernoulli, Beta, Categorical, Cauchy, ChiSquare,
    Dirichlet, Exponential, FisherSnedecor, Gamma,
    GumbelSoftmax, Gumbel, HalfCauchy, HalfNormal, HyperbolicSecant,
    Laplace, LogLaplace, LogCauchy, LogNormal, Logistic, Normal, Rayleigh,
    RelaxedBernoulli, StudentT, Uniform
]
@pytest.mark.parametrize("dist", dists_init)
def test_init(dist):
    model = dist()
    assert model.sample(1).shape[0] == 1
    assert model.sample(64).shape[0] == 64
    assert model.log_prob(model.sample(4)).shape[0] == 4
    model.get_parameters()


def test_generator():
    model = Generator()
    assert model.sample(64).shape == (64, 1)
    try:
        model.log_prob(model.sample(64))
    except NotImplementedError:
        pass
    model.get_parameters()


def test_distribution_errors():

    class Test(Distribution):

        def log_prob(self, value):
            pass

        def sample(self, batch_size):
            pass

    model = Test()

    methods = ['entropy', 'perplexity']
    methods_with_args = ['cdf', 'icdf', 'cross_entropy']
    properties = ['expectation', 'variance', 'median']

    for item in methods:
        try:
            getattr(model, item)()
        except NotImplementedError:
            pass
    for item in methods_with_args:
        try:
            getattr(model, item)(None)
        except NotImplementedError:
            pass
    for item in properties:
        try:
            getattr(model, item)()
        except NotImplementedError:
            pass

def test_normal_errors():
    model = Normal()
    assert model._diag_type == 'diag'
    model._diag_type = 'FAKE'
    try:
        model.log_prob(None)
    except NotImplementedError:
        pass
    try:
        model.sample(4)
    except NotImplementedError:
        pass
    try:
        model.entropy()
    except NotImplementedError:
        pass
    try:
        model.scale
    except NotImplementedError:
        pass

    model = Normal([0., 0.], [3., 1.0, 1., 3.])
    assert model._diag_type == 'cholesky'
    try:
        model.cdf(5.)
    except NotImplementedError:
        pass
    try:
        model.icdf(5.)
    except NotImplementedError:
        pass


# EOF
