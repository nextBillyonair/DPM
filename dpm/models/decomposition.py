import torch
from torch.nn import Parameter, init
from torch.nn import functional as F
from dpm.criterion import ELBO
from dpm.distributions import Distribution, Normal, Data
from dpm.train import train
from dpm.criterion import cross_entropy
import math

# spin off into class
def pca(X, k=2):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float()
    X = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(X)
    return torch.mm(X, V[:k].t())


class PCA():

    def __init__(self, k=2):
        self.k = k

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
        X = X - X.mean(dim=0, keepdim=True)
        self.U, self.S, self.V = torch.svd(X)
        self.eigen_values_ = self.S.pow(2)
        self.explained_variance_ = self.eigen_values_ / (X.shape[0] - 1)
        self.total_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_var

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
        return torch.mm(X, self.V[:self.k].t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def singular_values(self):
        return self.S[:self.k]

    @property
    def eigen_values(self):
        return self.eigen_values_[:self.k]

    @property
    def components(self):
        return self.V[:self.k]

    @property
    def explained_variance(self):
        return self.explained_variance_[:self.k]

    @property
    def explained_variance_ratio(self):
        return self.explained_variance_ratio_[:self.k]


class EMPPCA():

    def __init__(self, D=10, K=2):
        self.D = D
        self.K = K
        self.W = torch.Tensor(D, K).float()
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.W, _ = torch.qr(self.W)

    def fit(self, X, epochs=1000):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        X_t = X.t()
        for _ in range(epochs):
            Z = torch.inverse(self.W.t().mm(self.W)).mm(self.W.t()).mm(X_t)
            self.W = X_t.mm(Z.t()).mm(torch.inverse(Z.mm(Z.t())))
            self.W, _ = torch.qr(self.W)

    def transform(self, X):
        return X.mm(self.W)

    def reconstruct(self, Z):
        return Z.mm(torch.pinverse(self.W))


class PPCA_Variational(Distribution):

    def __init__(self, ppca):
        super().__init__()
        self.K = ppca.K
        self.D = ppca.D
        self.W = Parameter(torch.Tensor(ppca.K, ppca.D).float())
        self.noise = ppca.noise
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def sample(self, X, compute_logprob=False):
        dist = Normal(F.linear(X, self.W), self.noise * torch.eye(self.K), learnable=False)
        z = dist.sample(1).squeeze(0)
        if compute_logprob:
            return z, dist.log_prob(z)
        return z

    def log_prob(self, z, X):
        dist = Normal(self.W.mm(X), self.noise * torch.eye(self.D), learnable=False)
        return dist.log_prob(z)


class PPCA_Variational_V2(Distribution):

    def __init__(self, ppca):
        super().__init__()
        self.ppca = ppca

    def sample(self, X, compute_logprob=False):
        dist = Normal(F.linear(X, self.ppca.W.t()), self.ppca.noise * torch.eye(self.ppca.K), learnable=False)
        z = dist.sample(1).squeeze(0)
        if compute_logprob:
            return z, dist.log_prob(z)
        return z

    def log_prob(self, z, X):
        dist = Normal(self.ppca.W.t().mm(X), self.ppca.noise * torch.eye(self.ppca.D), learnable=False)
        return dist.log_prob(z)


# EXPERIMENTAL NOT DONE
class ProbabilisticPCA(Distribution):

    has_latents = True

    def __init__(self, D, K=2, noise=1., tau=None):
        super().__init__()
        self.D = D
        self.K = K
        self.W = Parameter(torch.Tensor(D, K).float())
        self.noise = torch.tensor(noise)
        self.latent = Normal(torch.zeros(K), torch.ones(K), learnable=False)
        self.tau = tau
        self.prior = None
        if tau:
            self.prior = Normal(torch.zeros(K), torch.full((K,), tau), learnable=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def prior_probability(self, z):
        if self.prior is None:
            return 0.
        return self.prior.log_prob(z)

    def log_prob(self, X, z):
        dist = Normal(F.linear(z, self.W), torch.full((z.size(0), self.D), self.noise), learnable=False)
        return dist.log_prob(X) + self.prior_probability(z)

    def sample(self, z=None, batch_size=1):
        if z is None:
            if self.prior is None:
                raise ValueError('PPCA has no prior distribution to sample latents from, please set tau in init')
            z = self.prior.sample(batch_size)
        dist = Normal(F.linear(z, self.W), torch.full((z.size(0), self.D), self.noise), learnable=False)
        return dist.sample(1).squeeze(0)

    def fit(self, X, variational_dist=None, elbo_kwargs={}, **kwargs):
        if variational_dist is None:
            variational_dist = PPCA_Variational_V2(self)

        data = Data(X)
        stats = train(data, self, ELBO(variational_dist, **elbo_kwargs), **kwargs)
        return stats

    def transform(self, X):
        return X.mm(self.W)

# EOF
