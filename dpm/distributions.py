from abc import abstractmethod, ABC
import torch
from torch import nn, distributions
from torch.nn import Module, Parameter, ModuleList
from torch.nn.functional import softplus
import numpy as np
import math

from dpm.transforms import Logit, Affine
import dpm.monte_carlo as monte_carlo

class Distribution(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_prob(self, value):
        raise NotImplementedError("log_prob method is not implemented")

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError("sample method is not implemented")

    def entropy(self, batch_size=10000):
        return -monte_carlo.monte_carlo(self.log_prob, self, batch_size)

    def perplexity(self, batch_size=10000):
        return self.entropy(batch_size).exp()

    def cross_entropy(self, model, batch_size=10000):
        return -model.log_prob(self.sample(batch_size)).mean()

    def softplus_inverse(self, value, threshold=20):
        inv = (value.exp() - 1.0).log()
        inv[value > threshold] = value[value > threshold]
        return inv

    def cdf(self, c, batch_size=10000):
        return monte_carlo.cdf(self, c, batch_size)

    def expectation(self, batch_size=10000):
        return monte_carlo.expectation(self, batch_size)

    def variance(self, batch_size=10000):
        return monte_carlo.variance(self, batch_size)

    def median(self, batch_size=10000):
        return monte_carlo.median(self, batch_size)


class Normal(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True, diag=False):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if diag:
            scale = torch.diag(scale)
        self.loc = loc
        self.cholesky_decomp = scale.view(self.n_dims, self.n_dims).cholesky()
        if learnable:
            self.loc = Parameter(self.loc)
            self.cholesky_decomp = Parameter(self.cholesky_decomp)

    def log_prob(self, value):
        model = distributions.MultivariateNormal(self.loc, self.scale)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.MultivariateNormal(self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return 0.5 * torch.logdet(2.0 * math.pi * math.e * self.scale)

    @property
    def scale(self):
        return torch.mm(self.cholesky_decomp, self.cholesky_decomp.t())

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Exponential(Distribution):

    def __init__(self, rate=1., learnable=True):
        super().__init__()
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate).view(-1)
        self.n_dims = len(rate)
        self._rate = self.softplus_inverse(rate)
        if learnable:
            self._rate = Parameter(self._rate)

    def log_prob(self, value):
        model = distributions.Exponential(self.rate)
        return model.log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        model = distributions.Exponential(self.rate)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return 1 - self.rate.log()

    @property
    def rate(self):
        return softplus(self._rate)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'rate':self.rate.item()}
        return {'rate':self.rate.detach().numpy()}


class GumbelSoftmax(Distribution):

    def __init__(self, probs=[0.5, 0.5], temperature=1.0,
                 hard=True, learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = temperature
        self.hard = hard
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = probs.log()
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = distributions.Categorical(probs=self.probs)
        return model.log_prob(value)

    def sample(self, batch_size):
        U = torch.rand((batch_size, self.n_dims))
        gumbel_samples = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = self.logits + gumbel_samples
        y = (y / self.temperature).softmax(dim=1)
        if self.hard:
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y = (y_hard - y).detach() + y
        return y

    @property
    def probs(self):
        return self.logits.softmax(dim=-1)

    def get_parameters(self):
        return {'probs':self.probs.detach().numpy()}


class Cauchy(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        model = distributions.Cauchy(self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Cauchy(self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return (4 * math.pi * self.scale).log()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Beta(Distribution):

    def __init__(self, alpha=0.5, beta=0.5, learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta).view(-1)
        self._alpha = self.softplus_inverse(alpha)
        self._beta = self.softplus_inverse(beta)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def log_prob(self, value):
        model = distributions.Beta(self.alpha, self.beta)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Beta(self.alpha, self.beta)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.Beta(self.alpha, self.beta).entropy()

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha': self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}


class LogNormal(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        model = distributions.LogNormal(self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.LogNormal(self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.LogNormal(self.loc, self.scale).entropy()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Gamma(Distribution):

    def __init__(self, alpha=1., beta=1., learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta).view(-1)
        self._alpha = self.softplus_inverse(alpha)
        self._beta = self.softplus_inverse(beta)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def log_prob(self, value):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.Gamma(self.alpha, self.beta).entropy()

    @property
    def alpha(self):
        return softplus(self._alpha)

    @property
    def beta(self):
        return softplus(self._beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}


class RelaxedBernoulli(Distribution):

    def __init__(self, probs=[0.5], temperature=1.0, learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = torch.tensor(temperature)
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = self.softplus_inverse(probs)
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.sample((batch_size,))

    @property
    def probs(self):
        return softplus(self.logits)

    def get_parameters(self):
        if self.n_dims == 1: return {'probs':self.probs.item()}
        return {'probs':self.probs.detach().numpy()}


class Uniform(Distribution):

    def __init__(self, low=0., high=1., learnable=True):
        super().__init__()
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low).view(-1)
        self.n_dims = len(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high).view(-1)
        self.alpha = low
        self.beta = high
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def log_prob(self, value):
        model = distributions.Uniform(self.low, self.high)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Uniform(self.low, self.high)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return (self.high - self.low).log()

    @property
    def low(self):
        return torch.min(self.alpha, self.beta)

    @property
    def high(self):
        return torch.max(self.alpha, self.beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'low':self.low.item(), 'high':self.high.item()}
        return {'low':self.low.detach().numpy(),
                'high':self.high.detach().numpy()}


class StudentT(Distribution):

    def __init__(self, df=1., loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        self._df = self.softplus_inverse(df)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)
            self._df = Parameter(self._df)

    def log_prob(self, value):
        model = distributions.StudentT(self.df, self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.StudentT(self.df, self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.StudentT(self.df, self.loc, self.scale).entropy()

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def df(self):
        return softplus(self._df)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc' : self.loc.item(), 'scale':self.scale.item(),
                    'df':self.df.item()}
        return {'loc' : self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy(),
                'df':self.df.detach().numpy()}


class Dirichlet(Distribution):

    def __init__(self, alpha=[0.5, 0.5], learnable=True):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).view(-1)
        self.n_dims = len(alpha)
        self._alpha = self.softplus_inverse(alpha)
        if learnable:
            self._alpha = Parameter(self._alpha)

    def log_prob(self, value):
        model = distributions.Dirichlet(self.alpha)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.Dirichlet(self.alpha)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.Dirichlet(self.alpha).entropy()

    @property
    def alpha(self):
        return softplus(self._alpha)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item()}
        return {'alpha':self.alpha.detach().numpy()}


class FisherSnedecor(Distribution):

    def __init__(self, df_1=1., df_2=1., learnable=True):
        super().__init__()
        if not isinstance(df_1, torch.Tensor):
            df_1 = torch.tensor(df_1).view(-1)
        self.n_dims = len(df_1)
        if not isinstance(df_2, torch.Tensor):
            df_2 = torch.tensor(df_2).view(-1)
        self._df_1 = self.softplus_inverse(df_1)
        self._df_2 = self.softplus_inverse(df_2)
        if learnable:
            self._df_1 = Parameter(self._df_1)
            self._df_2 = Parameter(self._df_2)

    def log_prob(self, value):
        model = distributions.FisherSnedecor(self.df_1, self.df_2)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.FisherSnedecor(self.df_1, self.df_2)
        return model.rsample((batch_size,))

    @property
    def df_1(self):
        return softplus(self._df_1)

    @property
    def df_2(self):
        return softplus(self._df_2)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'df_1':self.df_1.item(),'df_2':self.df_2.item()}
        return {'df_1':self.df_1.detach().numpy(),
                'df_2':self.df_2.detach().numpy()}


class HalfCauchy(Distribution):

    def __init__(self, scale=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.n_dims = len(scale)
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        model = distributions.HalfCauchy(self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.HalfCauchy(self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return distributions.HalfCauchy(self.scale).entropy()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'scale':self.scale.item()}
        return {'scale':self.scale.detach().numpy()}


class HalfNormal(Distribution):

    def __init__(self, scale=1., learnable=True):
        super().__init__()
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.n_dims = len(scale)
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        model = distributions.HalfNormal(self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.HalfNormal(self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return 0.5 * (0.5 * math.pi * self.scale.pow(2)).log() + 0.5

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'scale':self.scale.item()}
        return {'scale':self.scale.detach().numpy()}


class Laplace(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        model = distributions.Laplace(self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Laplace(self.loc, self.scale)
        return model.rsample((batch_size,))

    def entropy(self, batch_size=None):
        return (2 * self.scale * math.e).log()

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class DiracDelta(Distribution):

    def __init__(self, loc=0., learnable=False):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
            if len(loc.shape) == 0:
                loc = loc.view(-1)
        self.n_dims = loc.shape
        self.loc = loc

    def log_prob(self, value):
        raise NotImplementedError("Dirac Delta log_prob not implemented")

    def sample(self, batch_size):
        return self.loc.expand(batch_size, *self.n_dims)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item()}
        return {'loc':self.loc.detach().numpy()}


class Data(Distribution):

    def __init__(self, data, learnable=False):
        super().__init__()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.n_dims = data.size(-1)
        self.n_samples = len(data)
        self.data = data

    def log_prob(self, value):
        raise NotImplementedError("Data Distribution log_prob not implemented")

    def sample(self, batch_size):
        idx = torch.tensor(np.random.choice(self.data.size(0), size=batch_size))
        return self.data[idx]

    def get_parameters(self):
        return {'n_dims':self.n_dims, 'n_samples':self.n_samples}



# For ELBO!
# AKA Conditional Model
class ConditionalModel(Distribution):

    def __init__(self, input_dim, hidden_sizes, activation,
                 output_shapes, output_activations, distribution):
        super().__init__()
        prev_size = input_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(getattr(nn, activation)())
            layers.append(nn.BatchNorm1d(h))
            prev_size = h

        self.model = nn.Sequential(*layers)
        self.output_layers = nn.ModuleList()
        for output_shape, output_activation in zip(output_shapes, output_activations):
            layers = [nn.Linear(prev_size, output_shape)]
            if output_activation is not None:
                layers.append(getattr(nn, output_activation)())
            self.output_layers.append(nn.Sequential(*layers))

        self.distribution = distribution

    def _create_dist(self, x):
        h = self.model(x)
        dist_params = [output_layer(h) for output_layer in self.output_layers]
        return self.distribution(*dist_params, learnable=False)

    def sample(self, x, compute_logprob=False):
        # sample from q(z|x)
        dist = self._create_dist(x)
        z = dist.sample(1).squeeze(0)
        if compute_logprob:
            return z, dist.log_prob(z)
        return z

    def log_prob(self, z, x):
        # log_prob of q(z|x)
        return self._create_dist(x).log_prob(z)


class Generator(Distribution):

    def __init__(self, latent_distribution=None, input_dim=8,
                 hidden_sizes=[24, 24], activation="LeakyReLU",
                 output_shapes=[1], output_activations=[None]):
        super().__init__()
        self.latent_distribution = latent_distribution
        if latent_distribution is None:
            self.latent_distribution = Normal(torch.zeros(8), torch.eye(8), learnable=False)
        self.conditional_model = ConditionalModel(input_dim, hidden_sizes, activation,
                                                  output_shapes, output_activations, DiracDelta)
        self.n_dims = output_shapes[0]

    def log_prob(self, value):
        raise NotImplementedError("Generator log_prob not implemented")

    def sample(self, batch_size):
        latent_samples = self.latent_distribution.sample(batch_size)
        return self.conditional_model.sample(latent_samples)

    def get_parameters(self):
        return {'latent':self.latent_distribution.get_parameters()}


# Uses dist + transforms
class TransformDistribution(Distribution):

    def __init__(self, distribution, transforms, learnable=False):
        super().__init__()
        self.n_dims = distribution.n_dims
        self.distribution = distribution
        self.transforms = ModuleList(transforms)

    def log_prob(self, value):
        prev_value = value
        log_det = 0.0
        for transform in self.transforms[::-1]:
            value = transform.inverse(prev_value)
            log_det += transform.log_abs_det_jacobian(value, prev_value)
            prev_value = value
        return -log_det.sum(1) + self.distribution.log_prob(value)

    def sample(self, batch_size):
        samples = self.distribution.sample(batch_size)
        for transform in self.transforms:
            samples = transform(samples)
        return samples

    def get_parameters(self):
        return {'distribution':self.distribution.get_parameters(),
                'transforms': [transform.get_parameters()
                               for transform in self.transforms]}


class Convolution(Distribution):
    def __init__(self, models, learnable=False):
        super().__init__()
        self.n_dims = models[0].n_dims
        self.models = ModuleList(models)
        self.n_models = len(models)

    def log_prob(self, value):
        raise NotImplementedError("Convolution log_prob not implemented")

    def sample(self, batch_size):
        samples = torch.stack([sub_model.sample(batch_size)
                               for sub_model in self.models])
        return samples.sum(0)

    def get_parameters(self):
        return {'models':[model.get_parameters() for model in self.models]}


# Inherit Gamma?
class ChiSquare(Distribution):

    def __init__(self, df=1., learnable=True):
        super().__init__()
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df).view(-1)
        self._df = self.softplus_inverse(df)
        self.n_dims = len(df)
        if learnable:
            self._df = Parameter(self._df)

    def log_prob(self, value):
        alpha = 0.5 * self.df
        beta = torch.zeros_like(alpha).fill_(0.5)
        model = distributions.Gamma(alpha, beta)
        return model.log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        alpha = 0.5 * self.df
        beta = torch.zeros_like(alpha).fill_(0.5)
        model = distributions.Gamma(alpha, beta)
        return model.rsample((batch_size,))

    @property
    def df(self):
        return softplus(self._df)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'df':self.df.item()}
        return {'df':self.df.detach().numpy()}


class Logistic(Distribution):

    def __init__(self, loc=0., scale=1., learnable=True):
        super().__init__()
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc).view(-1)
        self.n_dims = len(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale).view(-1)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)

    def log_prob(self, value):
        zero = torch.zeros_like(self.loc)
        one = torch.ones_like(self.loc)
        model = TransformDistribution(Uniform(zero, one, learnable=False),
                                      [Logit(),
                                       Affine(self.loc, self.scale, learnable=False)])
        return model.log_prob(value)

    def sample(self, batch_size):
        zero = torch.zeros_like(self.loc)
        one = torch.ones_like(self.loc)
        model = TransformDistribution(Uniform(zero, one, learnable=False),
                                      [Logit(),
                                       Affine(self.loc, self.scale, learnable=False)])
        return model.sample(batch_size)

    @property
    def scale(self):
        return softplus(self._scale)

    def entropy(self, batch_size=None):
        return self.scale.log() + 2.

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Arcsine(Distribution):

    def __init__(self, low=0., high=1., learnable=True):
        super().__init__()
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low).view(-1)
        self.n_dims = len(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high).view(-1)
        self.alpha = low
        self.beta = high
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def log_prob(self, value):
        return - (math.pi * ((value - self.low) * (self.high - value)).sqrt()).log().reshape(-1)

    def sample(self, batch_size):
        raise NotImplementedError("sample not implemented")

    @property
    def low(self):
        return torch.min(self.alpha, self.beta)

    @property
    def high(self):
        return torch.max(self.alpha, self.beta)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'low':self.low.item(), 'high':self.high.item()}
        return {'low':self.low.detach().numpy(),
                'high':self.high.detach().numpy()}

#
# EOF
