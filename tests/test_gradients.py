import dpm.utils as newton
import torch
import pytest

# 1 d
def test_gradient():
    x = torch.tensor(3.).float()
    x.requires_grad = True

    y = x.pow(2)
    assert newton.grad(y, x) == 6.

    y = torch.tanh(x)
    assert newton.grad(y, x) - 0.009866 < 1e-3

    y = x.pow(2) * torch.tanh(x) * x.exp()
    assert newton.grad(y, x) - 301.577 < 0.1

    y = x.pow(2)
    assert newton.gradient(y, x) == 6.

    y = torch.tanh(x)
    assert newton.gradient(y, x) - 0.009866 < 1e-3

    y = x.pow(2) * torch.tanh(x) * x.exp()
    assert newton.gradient(y, x) - 301.577 < 0.1


def test_jacobian():
    x = torch.tensor([3., 4.])
    x.requires_grad = True
    y1 = x[0].pow(2) * x[1]
    y2 = 5*x[0] + x[1].sin()
    y = torch.cat((y1.view(-1, 1), y2.view(-1, 1)))
    jac = newton.gradient(y, x)
    assert (jac - torch.tensor([[24., 9.], [5., -0.6536]]) < 1e-2).all()

    x = torch.tensor([1., 2., 3.])
    x.requires_grad = True
    y1 = x[0]
    y2 = 5 * x[2]
    y3 = 4 * x[1].pow(2) - 2 * x[2]
    y4 = x[2] * x[0].sin()
    answer = torch.tensor([[1, 0, 0], [0, 0, 5], [0, 16, -2], [1.6209, 0, 0.8415]])
    y = torch.cat((y1.view(-1, 1), y2.view(-1, 1), y3.view(-1, 1), y4.view(-1, 1)))
    assert (newton.gradient(y, x) - answer < 1e-2).all()


def test_hessian_scalar():
    x = torch.tensor([1., 2., 3.])
    x.requires_grad = True
    y = x.dot(x)
    answer = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    assert (newton.hessian(y, x) - answer < 1e-2).all()

    y = (x.dot(x) * x).sum()
    answer = torch.tensor([[16.,  6.,  8.], [ 6., 20., 10.], [ 8., 10., 24.]])
    assert (newton.hessian(y, x) - answer < 1e-2).all()


def test_hessian_vector():
    x = torch.tensor([1., 2., 3.])
    x.requires_grad = True
    answer = torch.tensor([[[ 6.,  4.,  6.], [ 4.,  2.,  0.], [ 6.,  0.,  2.]],
                           [[ 4.,  2.,  0.], [ 2., 12.,  6.], [ 0.,  6.,  4.]],
                           [[ 6.,  0.,  2.], [ 0.,  6.,  4.], [ 2.,  4., 18.]]])
    y = (x.dot(x) * x)
    assert (newton.hessian(y, x) - answer < 1e-2).all()

# work out example
def test_newton():
    x = torch.tensor([1., 2., 3.])
    x.requires_grad = True
    y = (x.dot(x) * x).sum()
    assert newton.newton_step(y, x).shape == x.shape
    assert newton.newton_step(y, x, use_pinv=True).shape == x.shape





# EOF
