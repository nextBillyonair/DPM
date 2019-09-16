from dpm.distributions import (
    Distribution, ConditionalModel,
    Normal, Bernoulli, Data
)
from functools import partial
from dpm.utils import Sigmoid
from dpm.train import train
from dpm.criterion import cross_entropy, ELBO
import torch

# Pass Data to encoder -> mu, sigma
# Sample from latent(mu, sigma)
# Pass z to decoder -> Get Reconstruction

# Maximize reconstruction Prob


class VAE(Distribution):

    has_latents = True

    def __init__(self, encoder_args={}, decoder_args={}, prior=None, elbo_kwargs={}):
        super().__init__()
        preset_encoder_args={'input_dim':1, 'hidden_sizes':[24, 24],
                             'activation':'ReLU', 'output_shapes':[1, 1],
                             'output_activations':[None, 'Softplus'],
                             'distribution':partial(Normal, learnable=False)}
        preset_decoder_args={'input_dim':1, 'hidden_sizes':[24, 24],
                             'activation':'ReLU', 'output_shapes':[1],
                             'output_activations':[Sigmoid()],
                             'distribution':partial(Bernoulli, learnable=False)}

        preset_encoder_args.update(encoder_args)
        preset_decoder_args.update(decoder_args)

        self.encoder = ConditionalModel(**preset_encoder_args)
        self.decoder = ConditionalModel(**preset_decoder_args)
        self.criterion = ELBO(self.encoder, **elbo_kwargs)

        self.prior = prior
        if prior is None:
            latent_dim = preset_decoder_args['input_dim']
            self.prior = Normal(torch.zeros(latent_dim),
                                torch.ones(latent_dim),
                                learnable=False)


    def log_prob(self, X, Z=None):
        # latent given
        if Z is not None:
            return self.decoder.log_prob(X, Z) + self.prior.log_prob(Z)

        Z, encoder_probs = self.encoder.sample(X, compute_logprob=True)
        prior_probs = self.prior.log_prob(Z)
        decoder_log_probs = self.decoder.log_prob(X, Z)
        return decoder_log_probs + prior_probs - encoder_probs


    def sample(self, batch_size, compute_logprob=False):
        Z = self.prior.sample(batch_size)
        return self.decoder.sample(Z, compute_logprob)


    def fit(self, x, use_elbo=True, **kwargs):
        data = Data(x)
        if use_elbo:
            return train(data, self, self.criterion, **kwargs)
        return train(data, self, cross_entropy, **kwargs)

    def parameters(self):
        for name, param in self.named_parameters(recurse=True):
            if 'encoder' in name:
                continue
            yield param


# Normalize Data with X - Mu / Std For Normal Decoder
# Min Max Scaler for Bernoulli

# Note on decoder args:
# Gaussian -> 2 output shapes, activations: [None, SafeSoftplus()], distribtuion: Normal
# Bernoulli -> 1 output shape, activation: [Softmax()], distribtuion: Bernoulli
