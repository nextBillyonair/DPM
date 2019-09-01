from dpm.distributions import (
    Normal, Exponential, GumbelSoftmax, Cauchy,
    Beta, LogNormal, Gamma, RelaxedBernoulli, Uniform, StudentT,
    Dirichlet, FisherSnedecor, HalfCauchy, HalfNormal, Laplace,
)
from dpm.mixture_models import (
    MixtureModel, GumbelMixtureModel
)
from dpm.monte_carlo import (
    metropolis, metropolis_hastings,
    metropolis_langevin, hamiltonian_monte_carlo
)
import pytest

def test_mcmc_2d():
    true = MixtureModel([Normal([5.2, 5.2], [[3.0, 0.0], [0.0, 3.0]]),
                    Normal([0.0, 0.0], [[2.0, 0.0], [0.0, 2.0]]),
                    Normal([-5.2, -5.2], [[1.5, 0.0], [0.0, 1.5]])],
                    [0.25, 0.5, 0.25])

    samples = metropolis(true, epochs=100, burn_in=10)
    samples = metropolis(true, epochs=100, burn_in=10, keep_every=5)
    samples = metropolis(true, epochs=10, burn_in=1, keep_every=5, init=None)

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

    (Normal([0., 0.], [1., 0., 0., 1.0]), 2),
    (Exponential([0.5, 1.0]), 2),
    (Cauchy([0., 0.], [1., 1.]), 2),
    (Beta([0.5, 0.5], [1., 1.]), 2),
    (LogNormal([0., 0.], [1., 1.]), 2),
    (Gamma([0.5, 0.5], [1., 1.]), 2),
    (RelaxedBernoulli([0.5, 0.5]), 2),
    (Uniform([-1., -2.], [3., 4.]), 2),
    (StudentT([30., 10.], [1., 2.], [3., 4.]), 2),
    (Dirichlet([0.5, 1.5]), 2),
    (FisherSnedecor([10., 11.], [10., 11.]), 2),
    (HalfCauchy([1., 2.]), 2),
    (HalfNormal([1., 2.]), 2),
    (Laplace([0., 0.], [1., 1.]), 2),
    (MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75]), 2),
    (GumbelMixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75]), 2),
    (GumbelMixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75], hard=False), 2),
]

@pytest.mark.parametrize("dist,n_dims", test_dists)
def test_mcmc_complete(dist, n_dims):
    samples = metropolis(dist, epochs=10, burn_in=1, keep_every=5)
    if dist.n_dims == 1:
        samples = metropolis(dist, epochs=3, burn_in=1, keep_every=1, init=4.)
        samples = metropolis(dist, epochs=3, burn_in=1, keep_every=1, init=None)
        samples = metropolis_langevin(dist, epochs=3, burn_in=1, keep_every=1, init=4.)
        samples = metropolis_langevin(dist, epochs=3, burn_in=1, keep_every=1, init=None)
        samples = hamiltonian_monte_carlo(dist, epochs=3, burn_in=1, keep_every=1, init=4.)
        samples = hamiltonian_monte_carlo(dist, epochs=3, burn_in=1, keep_every=1, init=None)
    else:
        samples = metropolis(dist, epochs=3, burn_in=1, keep_every=1, init=[4., 4.])
