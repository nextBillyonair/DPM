import torch
from torch.nn import Parameter, init
import math
from dpm.distributions import Normal, Distribution, Data
from dpm.train import train
from dpm.divergences import cross_entropy

# Probabilistic Matrix Factorization
class PMF(Distribution):

    def __init__(self, N, M, D=5, tau=None):
        super().__init__()
        self.N = N
        self.M = M
        self.D = D # latent

        self.U = Parameter(torch.Tensor(D, N).float())
        self.V = Parameter(torch.Tensor(D, M).float())

        if tau is None:
            self.prior = None
        else:
            self.prior = Normal(0., tau, learnable=False)

        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.U, a=math.sqrt(5))
        init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def prior_penalty(self):
        if not self.prior: return 0.
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.parameters()]).view(-1, 1)).sum()

    def reconstruct(self):
        return self.U.t().mm(self.V)

    def log_prob(self, R):
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R)
        R = R.view(-1, self.N * self.M).float()
        mean = self.reconstruct().view(-1)
        return Normal(mean, torch.ones_like(mean), learnable=False).log_prob(R) + self.prior_penalty()

    def sample(self, batch_size, noise_std=1.0):
        return self.reconstruct().expand((batch_size, self.N, self.M)) + noise_std * torch.randn((batch_size, self.N, self.M))

    def fit(self, R, **kwargs):
        data = Data(R.view(-1, self.N * self.M))
        stats = train(data, self, cross_entropy, **kwargs)
        return stats

    def mse(self, R):
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R)
        return (self.reconstruct() - R.float()).pow(2).mean()

    def mae(self, R):
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R)
        return (self.reconstruct() - R.float()).abs().mean()



#EOF
