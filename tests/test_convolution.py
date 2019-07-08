from dpm.convolution import Convolution
from dpm.distributions import Normal
import pytest

dists = [
    (Convolution([Normal(15., 1.0), Normal(-10., 1.0)]),1),
    (Convolution([Normal([0., 1.], [1., 1.], diag=True),
                  Normal([-10., 10.], [1.0, 1.0], diag=True)]), 2)
]
@pytest.mark.parametrize("dist,n_dims", dists)
def test_convolution(dist, n_dims):
    assert dist.sample(1).shape == (1, n_dims)
    assert dist.sample(64).shape == (64, n_dims)

    try:
        dist.log_prob(dist.sample(64))
    except NotImplementedError:
        pass

    dist.get_parameters()
