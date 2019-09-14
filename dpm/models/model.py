from abc import abstractmethod, ABC
import torch
from dpm.distributions import Distribution, Data, ConditionalModel
from dpm.train import train
from dpm.criterion import cross_entropy

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

    def __init__(self, input_dim=1, output_shapes=[1],
                 hidden_sizes=[24, 24],
                 activation='LeakyReLU',
                 output_activations=[None],
                 distribution=None, prior=None):
        super().__init__()
        if not isinstance(output_shapes, list):
            output_shapes = [output_shapes]
        if not isinstance(output_activations, list):
            output_activations = [output_activations]
            
        self.n_dims = output_shapes
        self.model = ConditionalModel(input_dim, hidden_sizes=hidden_sizes,
                            activation=activation,
                            output_shapes=output_shapes,
                            output_activations=output_activations,
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

    def __init__(self, input_dim=1, output_shapes=[1], output_activations=[None],
                 distribution=None, prior=None):
        super().__init__(input_dim=input_dim, output_shapes=output_shapes,
                         hidden_sizes=[], activation="",
                         output_activations=output_activations,
                         distribution=distribution, prior=prior)



# EOF
