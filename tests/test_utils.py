import pytest
import dpm.utils as utils
import torch
import torch.nn.functional as F
import math

def test_kron():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[0, 5], [6, 7]])
    ret = utils.kron(A, B)
    assert (ret == torch.tensor([[0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]])).all()

    A = torch.tensor([[1, -4, 7], [-2, 3, 3]])
    B = torch.tensor([[8, -9, -6, 5], [1, -3, -4, 7], [2, 8, -8, -3], [1, 2, -5, -1]])
    ret = utils.kron(A, B)
    true = torch.tensor([[  8,  -9,  -6,   5, -32,  36,  24, -20,  56, -63, -42,  35],
                         [  1,  -3,  -4,   7,  -4,  12,  16, -28,   7, -21, -28,  49],
                         [  2,   8,  -8,  -3,  -8, -32,  32,  12,  14,  56, -56, -21],
                         [  1,   2,  -5,  -1,  -4,  -8,  20,   4,   7,  14, -35,  -7],
                         [-16,  18,  12, -10,  24, -27, -18,  15,  24, -27, -18,  15],
                         [ -2,   6,   8, -14,   3,  -9, -12,  21,   3,  -9, -12,  21],
                         [ -4, -16,  16,   6,   6,  24, -24,  -9,   6,  24, -24,  -9],
                         [ -2,  -4,  10,   2,   3,   6, -15,  -3,   3,   6, -15,  -3]])
    assert (ret == true).all()


def test_log_logit_sigmoid():
    utils.log(torch.tensor(0.))
    assert utils.log(torch.tensor(math.e)) - 1. < 1e-2
    assert (utils.logit(torch.tensor(0.9)) - 2.197224577 < 1e-2)
    assert (utils.logit(torch.tensor(0.2)) + 1.38629436112 < 1e-2)

    layer = utils.Logit()
    assert (layer(torch.tensor(0.9)) - 2.197224577 < 1e-2)
    assert (layer(torch.tensor(0.2)) + 1.38629436112 < 1e-2)

    layer = utils.Sigmoid()
    assert (layer(torch.tensor(0.5)) - 0.6224593312018545646389 < 1e-2)
    assert (layer(torch.tensor(3.)) - 0.9525741268224332191212 < 1e-2)
    assert (layer(torch.tensor(-3.)) - 0.04742587317756678087885 < 1e-2)


def test_flatten_reshape():
    x = torch.randn((3, 4, 5))
    flat = utils.Flatten()
    reshape = utils.Reshape(4, 5)
    flat_x = flat(x)
    assert flat_x.shape == (3, 20)
    reshape_x = reshape(flat_x)
    assert reshape_x.shape == (3, 4, 5)


def test_vec():
    A = torch.tensor([[1, -4, 7], [-2, 3, 3]])
    assert (utils.vec(A) == torch.tensor([[1], [-2], [-4], [3], [7], [3]])).all()

def test_transpose():
    A = torch.tensor([[1, -4, 7], [-2, 3, 3]])
    assert (utils.transpose(A) == torch.tensor([[1, -2], [-4, 3], [7, 3]])).all()

def test_sum():
    A = torch.tensor([[1, -4, 7], [-2, 3, 3]])
    assert utils.sum(A) == 8

def test_col_row_sum():
    A = torch.tensor([[1, -4, 7], [-2, 3, 3]])
    assert (utils.col_sum(A) == torch.tensor([-1, -1, 10])).all()
    assert (utils.row_sum(A) == torch.tensor([[4], [4]])).all()

def test_matrix():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[0, 5], [6, 7]])
    assert (utils.mm(A, B) == A.mm(B)).all()
    assert (utils.hadamard(A, B) == A * B).all()

def test_vectors():
    A = torch.tensor([1, 2, 3, 4])
    B = torch.tensor([5, 6, 7, 8])
    assert utils.sum(A) == A.sum()
    assert utils.sum(B) == B.sum()
    assert utils.dot(A, B) == A.dot(B)
    assert (utils.outer_product(A, B) == A.view(-1, 1) * B.view(1, -1)).all()

def test_mv():
    mat = torch.randn(2, 3)
    vec = torch.randn(3)
    assert (utils.mv(mat, vec) == torch.mv(mat, vec)).all()

def test_diag():
    A = torch.tensor([[1, -4], [-2, 3]])
    assert (torch.diag(A) == utils.diag(A)).all()
    X = torch.randn(4, 5, 5)
    assert (utils.batch_diag(X) == torch.stack([torch.diag(x) for x in X])).all()

def test_bmm():
    X = torch.randn(4, 5, 5)
    Y = torch.randn(4, 5, 5)
    assert (utils.bmm(X, Y) == torch.bmm(X, Y)).all()
    X = torch.randn(4, 7, 5)
    Y = torch.randn(4, 5, 7)
    assert (utils.bmm(X, Y) == torch.bmm(X, Y)).all()

def test_bilinear():
    A = torch.randn(3,5,4)
    l = torch.randn(2,5)
    r = torch.randn(2,4)
    assert (utils.bilinear(l, A, r) == F.bilinear(l, r, A)).all()

# EOF
