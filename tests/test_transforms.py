from dpm.distributions import LogNormal, Normal, TransformDistribution, Logistic
from dpm.transforms import *
import dpm
import torch.autograd as autograd
import torch
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

import pytest

transforms_dist_list = [
    (TransformDistribution(Normal(0., 1.), [Exp()]), 1),
    (TransformDistribution(Normal([0., 0.], [1., 1.], diag=True), [Exp()]), 2),
    (TransformDistribution(Normal(0.0, 1.0), [Affine(1.0, 2.0)]), 1),
]

@pytest.mark.parametrize("transform,n_dims", transforms_dist_list)
def test_transform_dist(transform, n_dims):

    assert transform.sample(1).shape == (1, n_dims)
    assert transform.log_prob(transform.sample(1)).shape == (1, )

    samples = transform.sample(64)
    assert samples.shape == (64, n_dims)

    log_probs = transform.log_prob(samples)
    assert log_probs.shape == (64, )

    transform.get_parameters()
    try:
        transform.entropy()
    except NotImplementedError:
        pass


def test_normal_lognormal():
    model = LogNormal(0.0, 1.0)
    transform = TransformDistribution(Normal(0.0, 1.0), [Exp()])

    x = model.sample(4)
    assert torch.all(transform.log_prob(x)- model.log_prob(x) < 1e-5)

    x = transform.sample(4)
    assert torch.all(transform.log_prob(x)- model.log_prob(x) < 1e-5)

    transform.get_parameters()


def test_normal_affine():
    model = Normal(1.0, 4.0)
    transform = TransformDistribution(Normal(0.0, 1.0),
                                      [Affine(1.0, 2.0)])

    x = model.sample(4)
    assert torch.all(transform.log_prob(x)- model.log_prob(x) < 1e-5)

    x = transform.sample(4)
    assert torch.all(transform.log_prob(x)- model.log_prob(x) < 1e-5)

    transform.get_parameters()


def test_logistic():
    base_distribution = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=torch.tensor([2.]), scale=torch.tensor([1.]))]
    model = TransformedDistribution(base_distribution, transforms)
    transform = Logistic(2., 1.)

    x = model.sample((4,)).reshape(-1, 1)
    assert torch.all(transform.log_prob(x)- model.log_prob(x).view(-1) < 1e-4)

    x = transform.sample(4)
    assert x.shape == (4, 1)
    assert torch.all(transform.log_prob(x)- model.log_prob(x).view(-1) < 1e-4)

    x = transform.sample(1)
    assert x.shape == (1, 1)
    assert torch.all(transform.log_prob(x)- model.log_prob(x).view(-1) < 1e-4)

    transform.get_parameters()


transforms_list = [
    Affine(),
    Affine(1., 2.),
    Exp(),
    Log(),
    Expm1(),
    Gumbel(),
    Gumbel(1., 2.),
    Power(),
    Power(0.),
    Reciprocal(),
    Square(),
    Sigmoid(),
    Logit(),
    SinhArcsinh(),
    SinhArcsinh(1., 2.),
    Softplus(1.),
    Softplus(2.),
    Softplus(-1.),
    Softsign(),
    Tanh(),
]

@pytest.mark.parametrize("t_form", transforms_list)
def test_transforms(t_form):
    for i in [0.0, 0.5, 1.0, 5.0, -2.0]:
        if isinstance(t_form, Logit) and i in [5.0, -2.0]: continue
        if isinstance(t_form, Log) and i in [0.0, -2.0]: continue
        if isinstance(t_form, Square) and i == -2.0: continue
        if isinstance(t_form, Reciprocal) and i == 0.0: continue
        if isinstance(t_form, Power) and i == -2.0: continue

        x = torch.tensor([[i]])
        assert (t_form.inverse(t_form(x)) - x) < 1e-3
        t_form.get_parameters()

        x = torch.tensor([[i]])
        x.requires_grad=True

        y = t_form(x)
        div = torch.log(torch.abs(autograd.grad(y, x)[0]))
        ladj = t_form.log_abs_det_jacobian(x, y)

        if div.item() == float('inf'):
            assert ladj.item() == float('inf')
        elif div.item() == float('-inf'):
            assert ladj.item() == float('-inf')
        else:
            assert torch.all(div - ladj < 1e-5)





# EOF
