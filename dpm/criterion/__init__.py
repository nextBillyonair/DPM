from .adversarial_loss import (
    AdversarialLoss,
    GANLoss, MMGANLoss, WGANLoss, LSGANLoss
)
from .divergences import (
    cross_entropy, perplexity,
    forward_kl, reverse_kl, js_divergence
)
from .elbo import ELBO
from .emd import (
    emd,
    make_constraint_matrix, make_distance_matrix
)
