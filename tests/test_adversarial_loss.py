from dpm.train import train
from dpm.distributions import Normal
from dpm.distributions import MixtureModel
from dpm.criterion import GANLoss, WGANLoss, LSGANLoss, MMGANLoss
import pytest

gans_list = [
    GANLoss(1),
    GANLoss(2, [32, 32, 32, 32], activation='ReLU'),
    WGANLoss(1),
    LSGANLoss(1),
    MMGANLoss(1),
    GANLoss(1, grad_penalty=10.),
    GANLoss(2, [32, 32, 32, 32], activation='ReLU', grad_penalty=10.),
    WGANLoss(1, grad_penalty=10.),
    LSGANLoss(1, grad_penalty=10.),
    MMGANLoss(1, grad_penalty=10.),
    GANLoss(1, use_spectral_norm=True),
    GANLoss(2, [32, 32, 32, 32], activation='ReLU', use_spectral_norm=True),
    WGANLoss(1, use_spectral_norm=True),
    LSGANLoss(1, use_spectral_norm=True),
    MMGANLoss(1, use_spectral_norm=True),
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
    X = p_model.sample(100)
    gan.classify(X)

# EOF
