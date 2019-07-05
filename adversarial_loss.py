import torch
import torch.optim as optim
from torch.nn import Module, Sequential, Linear, LeakyReLU, BCEWithLogitsLoss

class GANLoss(Module):
    def __init__(self, n_dims, hidden_size=24):
        super().__init__()
        self.n_dims = n_dims
        self.hidden_size = hidden_size
        self.discriminator_model = Sequential(Linear(n_dims, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, 1))
        self.bce_loss = BCEWithLogitsLoss()
        self.d_optimizer = optim.RMSprop(self.discriminator_model.parameters(),
                                         lr=1e-3)

    def forward(self, p_model, q_model, batch_size=64):
        self.d_optimizer.zero_grad()

        p_samples = p_model.sample(batch_size)
        p_values = self.discriminator_model(p_samples)
        p_loss = self.bce_loss(p_values, torch.ones_like(p_values))

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples)
        q_loss = self.bce_loss(q_values, torch.zeros_like(p_values))

        loss = 0.5*(p_loss + q_loss)
        loss.backward()
        self.d_optimizer.step()

        q_samples = q_model.sample(batch_size)
        q_values = self.discriminator_model(q_samples)
        q_loss = self.bce_loss(q_values, torch.ones_like(p_values))

        return q_loss

class WGANLoss(Module):
    def __init__(self, n_dims, hidden_size=24):
        super().__init__()
        self.n_dims = n_dims
        self.hidden_size = hidden_size
        self.discriminator_model = Sequential(Linear(n_dims, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, 1))
        self.d_optimizer = optim.RMSprop(self.discriminator_model.parameters(),
                                         lr=1e-3)

    def forward(self, p_model, q_model, batch_size=64):
        self.d_optimizer.zero_grad()

        p_samples = p_model.sample(batch_size)
        p_loss = self.discriminator_model(p_samples).mean()

        q_samples = q_model.sample(batch_size)
        q_loss = self.discriminator_model(q_samples).mean()

        loss = (p_loss - q_loss)
        loss.backward()
        self.d_optimizer.step()

        q_samples = q_model.sample(batch_size)
        q_loss = self.discriminator_model(q_samples).mean()

        return q_loss

class LSGANLoss(Module):
    def __init__(self, n_dims, hidden_size=24):
        super().__init__()
        self.n_dims = n_dims
        self.hidden_size = hidden_size
        self.discriminator_model = Sequential(Linear(n_dims, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, hidden_size),
                                              LeakyReLU(),
                                              Linear(hidden_size, 1))
        self.d_optimizer = optim.RMSprop(self.discriminator_model.parameters(),
                                         lr=1e-3)

    def forward(self, p_model, q_model, batch_size=64):
        self.d_optimizer.zero_grad()

        p_samples = p_model.sample(batch_size)
        p_loss = (self.discriminator_model(p_samples) \
                  - torch.ones(batch_size)).pow(2).mean()

        q_samples = q_model.sample(batch_size)
        q_loss = (self.discriminator_model(q_samples)).pow(2).mean()

        loss = 0.5*(p_loss + q_loss)
        loss.backward()
        self.d_optimizer.step()

        q_samples = q_model.sample(batch_size)
        q_loss = 0.5*(self.discriminator_model(q_samples) \
                     - torch.ones(batch_size)).pow(2).mean()

        return q_loss
