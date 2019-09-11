from .distribution import Distribution
from dpm.transforms import Chain

# Uses dist + transforms
class TransformDistribution(Distribution):

    def __init__(self, distribution, transforms, learnable=False):
        super().__init__()
        self.n_dims = distribution.n_dims
        self.distribution = distribution
        if isinstance(transforms, list):
            transforms = Chain(transforms)
        self.transforms = transforms

    def log_prob(self, value):
        log_det, final_value = self.transforms.log_abs_det_jacobian(value)
        return self.distribution.log_prob(final_value) - log_det.sum(1)

    def sample(self, batch_size):
        samples = self.distribution.sample(batch_size)
        return self.transforms(samples)

    # might need monotonize?
    def cdf(self, value):
        value = self.transforms.inverse(value)
        return self.distribution.cdf(value)

    def icdf(self, value):
        value = self.distribution.icdf(value)
        return self.transforms(value)

    def get_parameters(self):
        return {'distribution':self.distribution.get_parameters(),
                'transforms': [transform.get_parameters()
                               for transform in self.transforms]}
