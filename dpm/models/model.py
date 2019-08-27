from abc import abstractmethod, ABC
import torch
from dpm.distributions import Distribution, Data, ConditionalModel
from dpm.train import train
from dpm.divergences import cross_entropy

def fit(x, y, model, criterion=cross_entropy, **kwargs):
    data = Data(x, y)
    stats = train(data, model, criterion, **kwargs)
    return stats

def predict(x, model, compute_logprob=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return model.sample(x.float(), compute_logprob)

def parameterize(x, model):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return model.model(x.float())


class Model(Distribution):

    def __init__(self):
        super().__init__()

    def fit(self, x, y, **kwargs):
        return fit(x, y, self, **kwargs)

    def predict(self, x):
        return predict(x, self)

    def forward(self, x):
        return parameterize(x, self)

# wraps Conditonal Model around interface to NN Model (Expands Linear to multiple layers)
class NeuralModel(Model):

    def __init__(self, input_dim=1, output_shape=1,
                 hidden_sizes=[24, 24],
                 activation='LeakyReLU',
                 output_activation=None,
                 distribution=None, prior=None):
        super().__init__()
        self.model = ConditionalModel(input_dim, hidden_sizes=hidden_sizes,
                            activation=activation,
                            output_shapes=[output_shape],
                            output_activations=[output_activation],
                            distribution=distribution)
        self.prior = prior

    def prior_penalty(self):
        return self.prior.log_prob(torch.cat([p.view(-1) for p in self.model.parameters()]).view(-1, 1)).sum()

    def log_prob(self, x, y):
        if self.prior:
            return self.model.log_prob(y, x) + self.prior_penalty()
        return self.model.log_prob(y, x)

    def sample(self, x, compute_logprob=False):
        return self.model.sample(x, compute_logprob)



class LinearModel(NeuralModel):

    def __init__(self, input_dim=1, output_shape=1, output_activation=None,
                 distribution=None, prior=None):
        super().__init__(input_dim=input_dim, output_shape=output_shape,
                         hidden_sizes=[], activation="",
                         output_activation=output_activation,
                         distribution=distribution, prior=prior)



# EOF
