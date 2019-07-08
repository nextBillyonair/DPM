from dpm.distributions import LogNormal, Normal, TransformDistribution
from dpm.transforms import *
import dpm
import torch.autograd as autograd
import torch
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
    transform.entropy()


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


transforms_list = [
    Affine(),
    Affine(1., 2.),
    Exp(),
    Expm1(),
    Gumbel(),
    Gumbel(1., 2.),
    Power(),
    Power(3.),
    Reciprocal(),
    Sigmoid(),
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
    for i in [0.0, 1.0, 5.0, -2.0]:
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
