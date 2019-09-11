from dpm.train import train, Statistics
from dpm.criterion import ELBO
from dpm.distributions import InfiniteMixtureModel
from dpm.distributions import StudentT, Gamma, Distribution, ConditionalModel
from dpm.criterion import forward_kl

def test_elbo():
    variational_model = ConditionalModel(
        input_dim=1,
        hidden_sizes=[16, 16],
        activation='ReLU',
        distribution=Gamma,
        output_shapes=[1, 1],
        output_activations=['Softplus', 'Softplus'],
    )

    p_model = InfiniteMixtureModel([90.0], [-10.0], [5.0])
    q_model = InfiniteMixtureModel([80.0], [-10.0], [5.0])

    elbo_loss = ELBO(variational_model, optimizer='Adamax', lr=0.01, num_iterations=1)

    stats = Statistics()

    stats = train(
        p_model,
        q_model,
        elbo_loss,
        optimizer="Adamax",
        lr=0.1,
        epochs=3,
        log_interval=None,
        batch_size=64,
        stats=stats,
    )
