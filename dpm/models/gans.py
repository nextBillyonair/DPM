from dpm.distributions import Generator, Distribution, Data
from dpm.adversarial_loss import (
    GANLoss, MMGANLoss, WGANLoss, LSGANLoss
)
from dpm.train import train

# BASE

class GenerativeAdversarialNetwork(Distribution):

    # define them here please
    def __init__(self, criterion, generator_args, criterion_args):
        super().__init__()
        self.model = Generator(**generator_args)
        self.criterion = criterion(**criterion_args)
        self.n_dims = self.model.n_dims

    def log_prob(self, value):
        return self.model.log_prob(value)

    def sample(self, batch_size):
        return self.model.sample(batch_size)

    def fit(self, x, **kwargs):
        data = Data(x)
        stats = train(data, self.model, self.criterion,
                      optimizer='RMSprop', track_parameters=False, **kwargs)
        return stats

    def predict(self, x):
        return self.criterion.classify(x)


################################################################################
# GAN
################################################################################

class GAN(GenerativeAdversarialNetwork):

    def __init__(self, generator_args, criterion_args):
        super().__init__(GANLoss, generator_args, criterion_args)

class MMGAN(GenerativeAdversarialNetwork):

    def __init__(self, generator_args, criterion_args):
        super().__init__(MMGANLoss, generator_args, criterion_args)

class WGAN(GenerativeAdversarialNetwork):

    def __init__(self, generator_args, criterion_args):
        super().__init__(WGANLoss, generator_args, criterion_args)

class LSGAN(GenerativeAdversarialNetwork):

    def __init__(self, generator_args, criterion_args):
        super().__init__(LSGANLoss, generator_args, criterion_args)
