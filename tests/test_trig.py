import dpm.utils as utils
import math
import torch

def test_regular_trig():
    radian = utils.pi / 4.
    assert utils.sin(radian) - math.sqrt(2) / 2. < 1e-2
    assert utils.cos(radian) - math.sqrt(2) / 2. < 1e-2
    assert utils.tan(radian) - 1. < 1e-2
    assert utils.cot(radian) - 1. < 1e-2
    assert utils.sec(radian) - math.sqrt(2) < 1e-2
    assert utils.csc(radian) - math.sqrt(2) < 1e-2

    radian = utils.pi / 3.
    assert utils.sin(radian) - math.sqrt(3) / 2. < 1e-2
    assert utils.cos(radian) - 0.5 < 1e-2
    assert utils.tan(radian) - math.sqrt(3.) < 1e-2
    assert utils.cot(radian) - math.sqrt(3) / 3. < 1e-2
    assert utils.sec(radian) - 2. < 1e-2
    assert utils.csc(radian) - 2 * math.sqrt(3) / 3. < 1e-2


def test_hyper_trig():
    radian = utils.pi / 4.
    assert utils.sinh(radian) - 0.8686709615 < 1e-2
    assert utils.cosh(radian) - 1.3246090893 < 1e-2
    assert utils.tanh(radian) - 0.6557942026 < 1e-2
    assert utils.coth(radian) - 1.5248686188 < 1e-2
    assert utils.sech(radian) - 0.7549397087 < 1e-2
    assert utils.csch(radian) - 1.1511838709 < 1e-2


def test_inv_hyper_trig():
    assert utils.arcsinh(torch.tensor(1.)) - 0.881373587 < 1e-2
    assert utils.arccosh(torch.tensor(2.)) - 1.316957896 < 1e-2
    assert utils.arctanh(torch.tensor(0.6)) - 0.693147180 < 1e-2
    assert utils.arccoth(torch.tensor(2.)) - 0.5493061443 < 1e-2
    assert utils.arcsech(torch.tensor(0.6)) - 1.0986122886 < 1e-2
    assert utils.arccsch(torch.tensor(0.6)) - 1.2837956627 < 1e-2


def test_inve_trig():
    radian = utils.pi / 3.
    assert utils.arcsin(torch.tensor(1.)) - 1.5707963267 < 1e-2
    assert utils.arccos(torch.tensor(-1.)) - 3.141592653 < 1e-2
    assert utils.arctan(torch.tensor(2.)) - 1.10714871779 < 1e-2
    assert utils.arccot(torch.tensor(0.6)) - 1.0303768265 < 1e-2
    assert utils.arcsec(torch.tensor(2.)) - 1.0471975511 < 1e-2
    assert utils.arccsc(torch.tensor(2.)) - 0.523598775 < 1e-2

# EOF
