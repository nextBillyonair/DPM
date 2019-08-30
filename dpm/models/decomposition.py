import torch
from torch.nn import Parameter, init
from dpm.distributions import Distribution
from dpm.train import train
from dpm.divergences import cross_entropy

# spin off into class
def pca(X, k=2):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float()
    X = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(X.t())
    return torch.mm(X, V[:k].t())


class PCA():

    def __init__(self, k=2):
        self.k = k

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
        X = X - X.mean(dim=0, keepdim=True)
        self.U, self.S, self.V = torch.svd(X.t())
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

# EXPERIMENTAL NOT DONE
class ProbabilisticPCA(Distribution):

    def __init__(self, D=10, K=2, tau=None):
        self.K = K
        self.D
        self.W = Parameter(torch.Tensor(D, K).float())
        self.noise = Parameter(torch.tensor(1.))
        self.latent = Normal(torch.zeros(K), torch.ones(K), learnable=False)
        if tau:
            self.prior = Normal(0., tau, learnable=False)
        else:
            self.prior = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def prior_penalty(self):
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.parameters()]).view(-1, 1)).sum()

    def log_prob(self, value):
        var = self.W.mm(self.W.t()) + self.noise * torch.eyes(value.size(1))
        print(var.size())
        dist = Normal(torch.zeros(value.size(1)), var, learnable=False)
        if self.prior:
            return dist.log_prob(value) + self.prior_penalty()
        return dist.log_prob(value)

    def sample(self, batch_size):
        z = self.latent.sample(batch_size.size(0))
        dist = Normal(self.W.mv(z), self.noise * torch.eyes(self.D))
        return dist.sample(batch_size)

    def fit(self, X):
        data = Data(R.view(-1, self.N * self.M))
        stats = train(data, self, cross_entropy, **kwargs)
        return stats

    def transform(self, X):
        return X.mm(self.W)

# EOF
