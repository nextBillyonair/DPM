from abc import abstractmethod, ABC
import torch
from torch import nn, optim
from torch.nn.utils import spectral_norm
from torch.nn import Module, Sequential, Linear
import dpm.newton as newton

class AdversarialLoss(ABC, Module):

    def __init__(self, input_dim=1, hidden_sizes=[24, 24],
                 activation='LeakyReLU',
                 lr=1e-3, grad_penalty=None, use_spectral_norm=False):
        super().__init__()
        self.n_dims = input_dim
        prev_size = input_dim
        layers = []
        for h in hidden_sizes:
            if use_spectral_norm:
                layers.append(spectral_norm(Linear(prev_size, h)))
            else:
                layers.append(Linear(prev_size, h))
            layers.append(getattr(nn, activation)())
            prev_size = h

        if use_spectral_norm:
            layers.append(spectral_norm(Linear(prev_size, 1)))
        else:
            layers.append(Linear(prev_size, 1))

        self.discriminator_model = nn.Sequential(*layers)
        self.optimizer = optim.RMSprop(self.discriminator_model.parameters(),
                                       lr=lr)
        self.grad_penalty = grad_penalty

    @abstractmethod
    def discriminator_loss(self, p_values, q_values):
        raise NotImplementedError()

    @abstractmethod
    def generator_loss(self, q_values):
        raise NotImplementedError()

    def calculate_gp(self, p_samples, q_samples):
        eps = torch.rand((p_samples.shape))
        x_hat = eps * p_samples + (1 - eps) * q_samples
        x_hat.requires_grad = True
        x_hat_values = self.discriminator_model(x_hat).mean()
        gp = newton.gradient(x_hat_values, x_hat)
        penalty = self.grad_penalty * (torch.norm(gp, p=2) - 1).pow(2)
        return penalty

    def forward(self, p_model, q_model, batch_size=64):
        self.optimizer.zero_grad()

        p_samples = p_model.sample(batch_size)
        p_values = self.discriminator_model(p_samples.detach())

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples.detach())

        d_loss = self.discriminator_loss(p_values, q_values)

        if self.grad_penalty:
            d_loss += self.calculate_gp(p_samples.detach(), q_samples.detach())

        d_loss.backward()
        self.optimizer.step()

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples)
        return self.generator_loss(q_values)

    def classify(self, values):
        return self.discriminator_model(values)


# EOF
