from .adversarial_loss import (
    AdversarialLoss,
    GANLoss, MMGANLoss, WGANLoss, LSGANLoss
)
from .divergences import (
    cross_entropy, perplexity,
    forward_kl, reverse_kl,
    empirical_forward_kl, empirical_reverse_kl,
    js_divergence,
    exponential_divergence
)
from .elbo import ELBO
from .emd import (
    emd,
    make_constraint_matrix, make_distance_matrix
)
