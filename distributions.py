import torch
from torch.nn import Module, Parameter
import torch.distributions as distributions


class Distribution(Module):
    def __init__(self, **args):
        super().__init__()

    def log_prob(self, value):
        raise NotImplementedError("log_prob method is not implemented")

    def sample(self, batch_size):
        raise NotImplementedError("sample method is not implemented")


class Normal(Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.n_dims = len(loc)
        self.loc = Parameter(torch.tensor(loc))
        self.cholesky_decomp = Parameter(torch.tensor(scale).cholesky())

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
    def __init__(self, rate):
        super().__init__()
        self.n_dims = len(rate)
        self.log_rate = Parameter(torch.tensor(rate).log())

    def log_prob(self, value):
        model = distributions.Exponential(self.rate)
        return model.log_prob(value).sum(dim=-1)

    def sample(self, batch_size):
        model = distributions.Exponential(self.rate)
        return model.rsample((batch_size,))

    @property
    def rate(self):
        return self.log_rate.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'rate':self.rate.item()}
        return {'rate':self.rate.detach().numpy()}


class GumbelSoftmax(Distribution):
    def __init__(self, probs, temperature=1.0, hard=True):
        super().__init__()
        self.n_components = len(probs)
        self.temperature = temperature
        self.hard = hard
        self.logits = Parameter(torch.tensor(probs).log())

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
    def __init__(self, loc, scale):
        super().__init__()
        self.n_dims = len(loc)
        self.loc = Parameter(torch.tensor(loc))
        self.log_scale = Parameter(torch.tensor(scale).log())

    def log_prob(self, value):
        model = distributions.Cauchy(self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Cauchy(self.loc, self.scale)
        return model.rsample((batch_size,))

    @property
    def scale(self):
        return self.log_scale.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Beta(Distribution):
    def __init__(self, alpha, beta):
        super().__init__()
        self.n_dims = len(alpha)
        self.log_alpha = Parameter(torch.tensor(alpha).log())
        self.log_beta = Parameter(torch.tensor(beta).log())

    def log_prob(self, value):
        model = distributions.Beta(self.alpha, self.beta)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.Beta(self.alpha, self.beta)
        return model.rsample((batch_size,))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha': self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}


class LogNormal(Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.n_dims = len(loc)
        self.loc = Parameter(torch.tensor(loc))
        self.log_scale = Parameter(torch.tensor(scale).log())

    def log_prob(self, value):
        model = distributions.LogNormal(self.loc, self.scale)
        return model.log_prob(value).sum(-1)

    def sample(self, batch_size):
        model = distributions.LogNormal(self.loc, self.scale)
        return model.rsample((batch_size,))

    @property
    def scale(self):
        return self.log_scale.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'loc':self.loc.item(), 'scale':self.scale.item()}
        return {'loc':self.loc.detach().numpy(),
                'scale':self.scale.detach().numpy()}


class Gamma(Distribution):
    def __init__(self, alpha, beta):
        super().__init__()
        self.n_dims = len(alpha)
        self.log_alpha = Parameter(torch.tensor(alpha).log())
        self.log_beta = Parameter(torch.tensor(beta).log())

    def log_prob(self, value):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.Gamma(self.alpha, self.beta)
        return model.rsample((batch_size,))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def get_parameters(self):
        if self.n_dims == 1:
            return {'alpha':self.alpha.item(), 'beta':self.beta.item()}
        return {'alpha':self.alpha.detach().numpy(),
                'beta':self.beta.detach().numpy()}


class RelaxedBernoulli(Distribution):
    def __init__(self, probs, temperature=1.0):
        super().__init__()
        self.n_dims = len(probs)
        self.temperature = temperature
        self.logits = Parameter(torch.tensor(probs).log())

    def log_prob(self, value):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.log_prob(value)

    def sample(self, batch_size):
        model = distributions.RelaxedBernoulli(self.temperature, self.probs)
        return model.sample((batch_size,))

    @property
    def probs(self):
        return self.logits.softmax(dim=-1)

    def get_parameters(self):
        if self.n_dims == 1: return {'probs':self.props.item()}
        return {'probs':self.probs.detach().numpy()}


class Uniform(Distribution):
    def __init__(self, low, high):
        super().__init__()
        self.n_dims = len(low)
        self.alpha = Parameter(torch.tensor(low))
        self.beta = Parameter(torch.tensor(high))

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



# EOF
