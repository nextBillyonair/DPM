import torch
from dpm.distributions import (
    Bernoulli, Normal, Distribution,
    ConditionalModel, DiracDelta
)
from dpm.train import train
import numpy as np


class Encoder(Distribution):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ConditionalModel(**kwargs)

    def log_prob(self, z, x):
        return self.encoder.log_prob(z, x)

    def sample(self, x, compute_logprob=False):
        return self.encoder.sample(x, compute_logprob)



class Decoder(Distribution):

    def __init__(self, *args):
        super().__init__()
        self.decoder = ConditionalModel(*args)
        self.unit_normal = Normal([0.0], [1.0], learnable=False, diag=True)

    def log_prob(self, samples, latents=None):
        if latents is None:
            raise NotImplementedError("VAE Decoder log_prob not implemented without latents")
        return self.decoder.log_prob(samples, latents) + self.unit_normal.log_prob(latents)

    def sample(self, batch_size, compute_logprob=False):
        x = self.unit_normal.sample(batch_size)
        return self.decoder.sample(x, compute_logprob)


# FINISH
class VAE(Distribution):

    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, [24, 24], "ReLU",
                               [embedding_dim, embedding_dim],
                               [None, "Softplus"],
                                Normal)
        self.decoder = Decoder(embedding_dim, [24, 24], "ReLU",
                               [input_dim],
                               [None], DiracDelta)


    def encode(self, x):
        mu, sigma = self.encoder(x)
        return mu, sigma

    def sample_latent(self, mu, sigma):
        return Normal(mu, sigma).sample(1)

    def decoder(self, z):
        return self.decoder.sample(z)











# EOF
