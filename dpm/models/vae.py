import torch
from dpm.distributions import (
    Bernoulli, Normal, Distribution,
    ConditionalModel
)
from dpm.utils import Sigmoid
from dpm.train import train
import numpy as np

# VAE:
#   X -> Encoder -> Mu, Sigma
#   Sample
#   Sample -> Decoder -> Reonstruction
#

class Encoder(Distribution):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ConditionalModel(**kwargs)

    def forward(self, x):
        return self.encoder(x)

    def log_prob(self, z, x):
        return self.encoder.log_prob(z, x)

    def sample(self, x, compute_logprob=False):
        return self.encoder.sample(x, compute_logprob)



class Decoder(Distribution):

    def __init__(self, **kwargs):
        super().__init__()
        self.decoder = ConditionalModel(**kwargs)

    def log_prob(self, z, x):
        return self.decoder.log_prob(z, x)

    def sample(self, x, compute_logprob=False):
        return self.decoder.sample(x, compute_logprob)


# FINISH
class VAE(Distribution):

    def __init__(self, encoder_kwargs={}, decoder_kwargs={}):
        super().__init__()
        self.encoder = Encoder(**encoder_kwargs)
        self.decoder = Decoder(**decoder_kwargs)

        # self.encoder = Encoder(input_dim, [128, 64], "ReLU",
        #                        [embedding_dim, embedding_dim],
        #                        [None, "Softplus"],
        #                         Normal)
        # self.decoder = Decoder(embedding_dim, [24, 24], "ReLU",
        #                        [input_dim],
        #                        [Sigmoid()], output_dist)


    def encode(self, x):
        mu, sigma = self.encoder(x)
        return mu, sigma

    def sample_latent(self, mu, sigma):
        return Normal(mu, sigma, learnable=False).sample(1)

    def decoder(self, z):
        return self.decoder.sample(z)

    def forward(self, x):
        mu, sigma = self.encode(x)
        latent = self.sample_latent(mu, sigma)
        reconstruction, log_probs = self.decoder(latent, compute_logprob=True)
        return reconstruction, log_probs

    def log_prob(self, x):
        return self.forward(x)[1]


class BernoulliVAE(VAE):

    def __init__(self, encoder_kwargs={},
                 decoder_kwargs={
                    'output_shape'=[1],
                    'output_activation':[Sigmoid()],
                    'distribution':partial(Bernoulli, learnable=False)}):
        super().__init__(encoder_kwargs, decoder_kwargs)

# Regression
class NormalVAE(VAE):

    def __init__(self, encoder_kwargs={},
                 decoder_kwargs={}):
        super().__init__(encoder_kwargs, decoder_kwargs)






# EOF
