from dpm.train import train
from dpm.distributions import Normal
from dpm.mixture_models import MixtureModel
from dpm.adversarial_loss import GANLoss, WGANLoss, LSGANLoss, MMGANLoss
import pytest

gans_list = [
    GANLoss(1),
    GANLoss(2, [32, 32, 32, 32], activation='ReLU'),
    WGANLoss(1),
    LSGANLoss(1),
    MMGANLoss(1),
]

@pytest.mark.parametrize("gan", gans_list)
def test_forward(gan):
    if gan.n_dims == 1:
        q_model = MixtureModel([Normal([-0.5],[[1.0]]), Normal([0.5],[[1.0]])], [0.5, 0.5])
        p_model = MixtureModel([Normal([2.3], [[2.2]]), Normal([-2.3], [[2.2]])], [0.5, 0.5])
    else:
        q_model = MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75])
        p_model = MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75])

    gan(p_model, q_model)


@pytest.mark.parametrize("gan", gans_list)
def test_gan_train(gan):
    if gan.n_dims == 1:
        q_model = MixtureModel([Normal([-0.5],[[1.0]]), Normal([0.5],[[1.0]])], [0.5, 0.5])
        p_model = MixtureModel([Normal([2.3], [[2.2]]), Normal([-2.3], [[2.2]])], [0.5, 0.5])
    else:
        q_model = MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75])
        p_model = MixtureModel([Normal([0., 0.], [1., 0., 0., 1.0]), Normal([0., 0.], [1., 0., 0., 1.0])], [0.25, 0.75])

    train(p_model, q_model, gan, optimizer="RMSprop", epochs=3, lr=1e-3, batch_size=512)

# EOF
