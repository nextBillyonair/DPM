from dpm.visualize import (
    plot_stats, plot_models,
    plot_model, plot_hists,
    plot_hist, plot_mcmc,
    plot_loss_function, plot_emd_partition, plot_emd_gamma,
    plot_emd_hist, get_emd_colormap
)
from dpm.emd import emd, make_distance_matrix
from dpm.distributions import (
    Normal, Exponential, GumbelSoftmax, Cauchy,
    Beta, LogNormal, Gamma, RelaxedBernoulli, Uniform, StudentT,
    Dirichlet, FisherSnedecor, HalfCauchy, HalfNormal, Laplace,
    Logistic, ChiSquare
)
from dpm.mixture_models import MixtureModel, GumbelMixtureModel
from dpm.monte_carlo import metropolis_hastings
from dpm.train import train
from dpm.divergences import (
    forward_kl, reverse_kl, js_divergence, cross_entropy
)
import numpy as np
import matplotlib.pyplot as plt
import pytest

models = [
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

    (Normal([0., 0.], [1., 1.]), Normal([0., 0.], [1., 1.])),
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
    (GumbelMixtureModel([Normal([0., 0.], [1., 1.]), Normal([1., 1.], [3., 3.])], [0.25, 0.75]),
     GumbelMixtureModel([Normal([0., 0.], [1., 1.]), Normal([1., 1.], [3., 3.])], [0.25, 0.75])),
    (GumbelMixtureModel([Normal([0., 0.], [1., 1.]), Normal([1., 1.], [3., 3.])], [0.25, 0.75], hard=False),
     GumbelMixtureModel([Normal([0., 0.], [1., 1.]), Normal([1., 1.], [3., 3.])], [0.25, 0.75], hard=False)),
    (ChiSquare([4., 4.]), ChiSquare([4., 4.])),
    (Logistic([0., 0.], [1., 1.]), Logistic([0., 0.], [1., 1.])),
]

@pytest.mark.parametrize("p_model,q_model", models)
def test_plot_models(p_model, q_model):
    plot_models(p_model, q_model, batch_size=64, n_plot=50)
    plt.close()
    plot_model(p_model, batch_size=64)
    plt.close()



@pytest.mark.parametrize("p_model,q_model", models)
def test_plot_hists(p_model, q_model):
    p_samples = p_model.sample(64)
    q_samples = q_model.sample(64)
    assert p_samples.shape == (64, p_model.n_dims)
    assert q_samples.shape == (64, q_model.n_dims)
    plot_hist(p_samples.detach())
    plt.close()
    plot_hists(p_samples.detach(), q_samples.detach())
    plt.close()


stats = [
    (Normal(0., 1.), Normal(1., 1.), 2),
    (Exponential(1.), Exponential(0.5), 1),
    (Beta(0.5, 1.), Beta(3., 3.), 2),
    (StudentT(30.0, 1.0, 3.0), StudentT(40.0, 5.0, 5.0), 3),
]
@pytest.mark.parametrize("p_model,q_model,n_stats", stats)
def test_plot_stats(p_model, q_model, n_stats):
    stats = train(p_model, q_model, forward_kl, epochs=10)
    assert len(stats.data['loss']) == 10
    assert len(stats.data.keys()) == n_stats + 1
    for key in stats.data.keys():
        assert len(stats.data[key]) == 10
    if isinstance(p_model, Normal):
        plot_stats(stats, goals=[p_model.loc, p_model.scale])
    elif isinstance(p_model, Exponential):
        plot_stats(stats, goals=[p_model.rate])
    elif isinstance(p_model, Beta):
        plot_stats(stats, goals=[p_model.alpha, p_model.beta])
    elif isinstance(p_model, StudentT):
        plot_stats(stats, goals=[p_model.df, p_model.loc, p_model.scale])
    else:
        plot_stats(stats)
    plt.close()


mcmc_models = [
    (Normal(0., 1.), 1),
    (Normal([0., 0.], [1., 1.]), 2),
]
@pytest.mark.parametrize("dist,n_dims", mcmc_models)
def test_mcmc_plot(dist, n_dims):
    samples = metropolis_hastings(dist, epochs=10, burn_in=2, keep=2)
    assert samples.size(1) == n_dims
    plot_mcmc(samples)
    plt.close()


losses = [
    forward_kl, reverse_kl, js_divergence, cross_entropy,
]
@pytest.mark.parametrize("loss", losses)
def test_plot_loss(loss):
    plot_loss_function(loss, batch_size=16, n_plot=5)


def test_emd_plots_discrete():
    Q = np.array([12,7,4,1,19,14,9,6,3,2])
    P = np.array([1,5,11,17,13,9,6,4,3,2])

    Q = Q / np.sum(Q)
    P = P / np.sum(P)

    colormap = get_emd_colormap(vmax=max(len(P), len(Q)))
    plot_emd_hist(Q, title=r"Q", colorMap=colormap)
    plot_emd_hist(Q, title=r"Q")
    plot_emd_hist(P, title=r"P", colorMap=colormap)
    emd_primal, gamma_primal = emd(Q, P)
    plot_emd_gamma(gamma_primal)
    plot_emd_partition(gamma_primal, colormap)
    plot_emd_partition(gamma_primal.T, colormap)




# EOF
