from abc import abstractmethod, ABC
import torch
from torch import nn, optim
from torch.nn import (
    Module, Sequential, Linear, LeakyReLU,
    BCEWithLogitsLoss, MSELoss
)

class AdversarialLoss(ABC, Module):

    def __init__(self, input_dim, hidden_sizes=[24, 24],
                 activation='LeakyReLU',
                 lr=1e-3):
        super().__init__()
        self.n_dims = input_dim
        prev_size = input_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(getattr(nn, activation)())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.discriminator_model = nn.Sequential(*layers)
        self.optimizer = optim.RMSprop(self.discriminator_model.parameters(),
                                       lr=lr)

    @abstractmethod
    def discriminator_loss(self, p_values, q_values):
        raise NotImplementedError()

    @abstractmethod
    def generator_loss(self, q_values):
        raise NotImplementedError()

    def forward(self, p_model, q_model, batch_size=64):
        self.optimizer.zero_grad()

        p_samples = p_model.sample(batch_size)
        p_values = self.discriminator_model(p_samples.detach())

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples.detach())

        d_loss = self.discriminator_loss(p_values, q_values)
        d_loss.backward()
        self.optimizer.step()

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples)
        return self.generator_loss(q_values)

# Also NSGAN
class GANLoss(AdversarialLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_loss = BCEWithLogitsLoss()

    def discriminator_loss(self, p_values, q_values):
        p_loss = self.bce_loss(p_values, torch.ones_like(p_values))
        q_loss = self.bce_loss(q_values, torch.zeros_like(q_values))
        return p_loss + q_loss

    def generator_loss(self, q_values):
        return self.bce_loss(q_values, torch.ones_like(q_values))


class MMGANLoss(AdversarialLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_loss = BCEWithLogitsLoss()

    def discriminator_loss(self, p_values, q_values):
        p_loss = self.bce_loss(p_values, torch.ones_like(p_values))
        q_loss = self.bce_loss(q_values, torch.zeros_like(q_values))
        return p_loss + q_loss

    def generator_loss(self, q_values):
        return -self.bce_loss(q_values, torch.zeros_like(q_values))


class WGANLoss(AdversarialLoss):

    def discriminator_loss(self, p_values, q_values):
        return p_values.mean() - q_values.mean()

    def generator_loss(self, q_values):
        return q_values.mean()


class LSGANLoss(AdversarialLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = MSELoss()

    def discriminator_loss(self, p_values, q_values):
        p_loss = self.mse_loss(p_values, torch.ones_like(p_values))
        q_loss = self.mse_loss(q_values, torch.zeros_like(q_values))
        return p_loss + q_loss

    def generator_loss(self, q_values):
        return self.mse_loss(q_values, torch.ones_like(q_values))








# EOF
