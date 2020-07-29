from dpm.models import (
    OrdinalLayer, OrdinalModel,
    erf_cdf, exp_cdf, tanh_cdf,
    normal_cdf, laplace_cdf, cauchy_cdf,
    OrdinalLoss,
)
from dpm.visualize import (
    plot_ordinal_classes, plot_ordinal_classes_from_layer
)
import numpy as np
import torch
import torch.nn as nn
import pytest


def test_ordinal_construction():
    predictor = nn.Sequential(
        nn.Linear(1, 12),
        nn.ELU(),
        nn.Linear(12, 12),
        nn.ELU(),
        nn.Linear(12, 1, bias=False)
    )
    model = OrdinalModel(predictor, OrdinalLayer(5))
    assert len(model.ordinal.theta) == 4 # minus 1 from target
    assert len(model.ordinal.threshold) == 4

    # test sort order of threshold
    t = model.ordinal.threshold
    assert torch.all(t[1:] - t[:-1] > 0)

    f = torch.linspace(-12, 12, 100).view(-1, 1)
     # runs
    assert model(f).shape == (100, 5)
    assert model.ordinal(f).shape == (100, 5)

    # test prob property
    assert torch.all((model.ordinal(f).exp().sum(-1) - 1) < 1e-5)


test_funcs = [
    erf_cdf, exp_cdf, tanh_cdf,
    normal_cdf, laplace_cdf, cauchy_cdf
]
@pytest.mark.parametrize("func", test_funcs)
def test_func_versions(func):
    f = torch.linspace(-12, 12, 100).view(-1, 1)
    model = OrdinalLayer(5, func=func)

    assert len(model.theta) == 4 # minus 1 from target
    assert len(model.threshold) == 4
    assert model(f).shape == (100, 5)
    assert torch.all((model(f).exp().sum(-1) - 1) < 1e-5)


def test_ordinal_loss():
    # Define Models
    predictor = nn.Sequential(
        nn.Linear(1, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 1, bias=False)
    )

    f = torch.linspace(-5, 5, 100).view(-1, 1)
    model = OrdinalModel(predictor, OrdinalLayer(5))
    criterion = OrdinalLoss('mean')
    loss = criterion(model(f), torch.randint(5, (100, )))
    assert loss.shape == ()


def test_ordinal_plot():
    f = torch.linspace(-12, 12, 100)
    cutpoints = torch.tensor([-3, 3])
    plot_ordinal_classes(f, cutpoints, func=laplace_cdf, title='Laplace')

    model = OrdinalLayer(5, func=tanh_cdf)
    plot_ordinal_classes_from_layer(f, model)
