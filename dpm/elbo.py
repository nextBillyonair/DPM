import torch
from torch.nn import Module
from torch import optim

class ELBO(Module):
    def __init__(self, variational_model, lr=0.01, optimizer='Adam', num_iterations=1):
        super().__init__()
        self.variational_model = variational_model
        self.optimizer = getattr(optim, optimizer)(variational_model.parameters(), lr=lr)
        self.num_iterations = num_iterations

    def forward(self, p_model, q_model, batch_size=64):
        assert q_model.has_latents, "Error: Q Model does not have latent variables"

        p_samples = p_model.sample(batch_size)
        # Sample from q(z | x) and compute_logprob
        loss = 0
        for i in range(self.num_iterations):
            latents, variational_log_prob = self.variational_model.sample(p_samples, compute_logprob=True)
            # Compute log p(x, z) = log p(x | z) + log p(z)
            q_log_prob = q_model.log_prob(p_samples, latents)
            loss += -(q_log_prob - variational_log_prob).mean()

        return loss / self.num_iterations

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
