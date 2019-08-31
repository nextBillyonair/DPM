import pytest
import dpm.utils as utils
import torch
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


def test_non_trig_functions():
    utils.log(torch.tensor(0.))
    assert utils.log(torch.tensor(math.e)) - 1. < 1e-3
    assert (utils.logit(torch.tensor(0.9)) - 2.197224577 < 1e-3)
    assert (utils.logit(torch.tensor(0.2)) + 1.38629436112 < 1e-3)






# EOF
