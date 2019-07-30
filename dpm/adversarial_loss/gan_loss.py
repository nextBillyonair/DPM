from .adversarial_loss import AdversarialLoss
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch

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
