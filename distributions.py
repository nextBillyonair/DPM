from abc import abstractmethod, ABC
import torch
from torch import nn, distributions
from torch.nn import Module, Parameter
from torch.nn.functional import softplus

class Distribution(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_prob(self, value):
        raise NotImplementedError("log_prob method is not implemented")

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError("sample method is not implemented")

    def softplus_inverse(self, value, threshold=20):
        inv = (value.exp() - 1.0).log()
        inv[value > threshold] = value[value > threshold]
        return inv


class Normal(Distribution):

    def __init__(self, loc, scale, learnable=True):
        super().__init__()
        self.n_dims = len(loc)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
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

    @property
    def scale(self):
        return torch.mm(self.cholesky_decomp, self.cholesky_decomp.t())

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Exponential(Distribution):

    def __init__(self, rate, learnable=True):
        super().__init__()
        self.n_dims = len(rate)
        if not isinstance(rate, torch.Tensor):
            rate = torch.tensor(rate)
        self._rate = self.softplus_inverse(rate)
        if learnable:
            self._rate = Parameter(self._rate)

    def log_prob(self, value):
        model = distributions.Exponential(self.rate)
        return model.log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        model = distributions.Exponential(self.rate)
        return model.rsample((batch_size,))

    @property
    def rate(self):
        return softplus(self._rate)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'rate':self.rate.item()}
        return {'rate':self.rate.detach().numpy()}


class GumbelSoftmax(Distribution):

    def __init__(self, probs, temperature=1.0, hard=True, learnable=True):
        super().__init__()
        self.n_components = len(probs)
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
        U = torch.rand((batch_size, self.n_components))
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

    def __init__(self, loc, scale, learnable=True):
        super().__init__()
        self.n_dims = len(loc)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
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

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Beta(Distribution):

    def __init__(self, alpha, beta, learnable=True):
        super().__init__()
        self.n_dims = len(alpha)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta)
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

    def __init__(self, loc, scale, learnable=True):
        super().__init__()
        self.n_dims = len(loc)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
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

    @property
    def scale(self):
        return softplus(self._scale)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Gamma(Distribution):

    def __init__(self, alpha, beta, learnable=True):
        super().__init__()
        self.n_dims = len(alpha)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta)
        self._alpha = self.softplus_inverse(alpha)
        self._beta = self.softplus_inverse(beta)
        if learnable:
            self._alpha = Parameter(self._alpha)
            self._beta = Parameter(self._beta)

    def log_prob(self, value):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.rsample((batch_size,))

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

    def __init__(self, probs, temperature=1.0, learnable=True):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = temperature
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.logits = self.softplus_inverse(probs)
        if learnable:
            self.logits = Parameter(self.logits)

    def log_prob(self, value):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.sample((batch_size,))

    @property
    def probs(self):
        return softplus(self.logits)

    def get_parameters(self):
        if self.n_dims == 1: return {'probs':self.props.item()}
        return {'probs':self.probs.detach().numpy()}


class Uniform(Distribution):

    def __init__(self, low, high, learnable=True):
        super().__init__()
        self.n_dims = len(low)
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high)
        self.alpha = low
        self.beta = high
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def log_prob(self, value):
        model = distributions.Uniform(self.low, self.high)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.Uniform(self.low, self.high)
        return model.rsample((batch_size,))

    @property
    def low(self):
        if self.alpha <= self.beta:
            return self.alpha
        return self.beta

    @property
    def high(self):
        if self.alpha <= self.beta:
            return self.beta
        return self.alpha

    def get_parameters(self):
        if self.n_dims == 1:
            return {'low':self.low.item(), 'high':self.high.item()}
        return {'low':self.low.detach().numpy(),
                'high':self.high.detach().item()}


class StudentT(Distribution):

    def __init__(self, df, loc, scale, learnable=True):
        super().__init__()
        self.n_dims = len(loc)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df)
        self.loc = loc
        self._scale = self.softplus_inverse(scale)
        self._df = self.softplus_inverse(df)
        if learnable:
            self.loc = Parameter(self.loc)
            self._scale = Parameter(self._scale)
            self._df = Parameter(self._df)

    def log_prob(self, value):
        model = distributions.StudentT(self.df, self.loc, self.scale)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.StudentT(self.df, self.loc, self.scale)
        return model.rsample((batch_size,))

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

    def __init__(self, alpha, learnable=True):
        self.n_dims = len(alpha)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        self._alpha = self.softplus_inverse(alpha)
        if learnable:
            self._alpha = Parameter(self._alpha)

    def log_prob(self, value):
        model = distributions.Dirichlet(self.alpha)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.Dirichlet(self.alpha)
        return model.rsample((batch_size,))

    @property
    def alpha(self):
        return softplus(self._alpha)

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item()}
        return {'alpha':self.alpha.detach().numpy()}


class FisherSnedecor(Distribution):

    def __init__(self, df_1, df_2, learnable=True):
        self.n_dims = len(df_1)
        if not isinstance(df_1, torch.Tensor):
            df_1 = torch.tensor(df_1)
        if not isinstance(df_2, torch.Tensor):
            df_2 = torch.tensor(df_2)
        self._df_1 = self.softplus_inverse(df_1)
        self._df_2 = self.softplus_inverse(df_2)
        if learnable:
            self._df_1 = Parameter(self._df_1)
            self._df_2 = Parameter(self._df_2)

    def log_prob(self, value):
        model = distributions.FisherSnedecor(self.df_1, self.df_2)
        return model.log_prob(value)

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


# Missing: Half-Cauchy, Half Normal, Laplace


# For ELBO!
class VariationalModel(Distribution):

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
            layers = []
            layers.append(nn.Linear(prev_size, output_shape))
            if output_activation is not None:
                layers.append(getattr(nn, output_activation)())
            self.output_layers.append(nn.Sequential(*layers))

        self.distribution = distribution

    def _create_dist(self, x):
        h = self.model(x)
        dist_params = [output_layer(h) for output_layer in self.output_layers]
#         print(dist_params[0].mean(), dist_params[1].mean())
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



# EOF
