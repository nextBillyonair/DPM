from dpm.mixture_models import (
    MixtureModel, GumbelMixtureModel, InfiniteMixtureModel
)
from dpm.distributions import Normal
import numpy as np
import pytest

mm_models = [
    (MixtureModel([Normal(0.0, 1.0), Normal(0.0, 1.0)], [0.5, 0.5]), 1),
    (MixtureModel([Normal([0.0, 0.0], [1.0, 1.0], diag=True),
                   Normal([0.0, 0.0], [1.0, 0.0, 0.0, 1.0])], [0.5, 0.5]), 2),
    (GumbelMixtureModel([Normal(0.0, 1.0), Normal(0.0, 1.0)], [0.5, 0.5]), 1),
    (GumbelMixtureModel([Normal([0.0, 0.0], [1.0, 1.0], diag=True),
                         Normal([0.0, 0.0], [1.0, 0.0, 0.0, 1.0])], [0.5, 0.5]), 2),
    (GumbelMixtureModel([Normal(0.0, 1.0), Normal(0.0, 1.0)], [0.5, 0.5], hard=False), 1),
    (GumbelMixtureModel([Normal([0.0, 0.0], [1.0, 1.0], diag=True),
                         Normal([0.0, 0.0], [1.0, 0.0, 0.0, 1.0])], [0.5, 0.5], hard=False), 2),
]
@pytest.mark.parametrize("model,n_dims", mm_models)
def test_mixture_model(model, n_dims):
    assert model.sample(1).shape == (1, n_dims)
    assert model.sample(64).shape == (64, n_dims)

    assert model.log_prob(model.sample(1)).shape == (1, )
    assert model.log_prob(model.sample(64)).shape == (64, )

    assert (model.get_parameters()['probs'] == np.array([0.5, 0.5])).all()


imm_models = [
    InfiniteMixtureModel(10., 10., 1.)
]
@pytest.mark.parametrize("model", imm_models)
def test_mixture_model(model):
    assert model.sample(1).shape == (1, 1)
    assert model.sample(64).shape == (64, 1)

    assert model.log_prob(model.sample(1)).shape == (1, )
    assert model.log_prob(model.sample(64)).shape == (64, )

    assert (model.get_parameters()['probs'] == np.array([0.5, 0.5])).all()
