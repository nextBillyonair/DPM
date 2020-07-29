import torch
import torch.nn as nn
import torch.nn.functional as F

########################
# NOTE
# I'm not sure how to fit this in with Models,
# so it will be a sub file under models, with its
# own interface until a refactoring occurs
# Especially if you want to sample, but log_prob is just the prediction
########################

## Purpose: Reimplements below but without cutting gradients (trick: use modified cumsum)
# https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/


# Base Layer For Ordinal Prediction
class OrdinalLayer(nn.Module):
    def __init__(self, n_classes, func=torch.sigmoid):
        super().__init__()
        self.func = func
        self.theta = nn.Parameter(torch.linspace(-1, 1, n_classes - 1))
        self.mask = torch.tensor([1] + [0 for _ in range(n_classes - 2)])

    def forward(self, x):
        # Input: x -> (B, *, 1)
        size = x.size()
        x = self.threshold - x.view(-1, 1)

        x = torch.cat((
                torch.zeros(x.size(0), 1),
                self.func(x), # any cdf
                torch.ones(x.size(0), 1)
            ), dim=-1)

        x = x[:, 1:] - x[:, :-1]

        # directly give log probs,
        # use NLL bc they cant be softmaxed
        # Return: Log Probs
        # x -> (B, *, N_CLASSES)
        return (x + 1e-8).log().view(*size[:-1], -1)

    @property
    def threshold(self):
        return (self.theta * self.mask + F.softplus(self.theta) * (1 - self.mask)).cumsum(-1)



# Module Wrapper For Prediction To Make Full Ordinal Model
# Predictor is any NN that outputs single number
class OrdinalModel(nn.Module):
    def __init__(self, predictor, ordinal):
        super().__init__()
        # predictor -> Module that outputs (B, *, 1)
        self.predictor = predictor
        self.ordinal = ordinal

    def forward(self, x):
        return self.ordinal(self.predictor(x))


# Functions converted to support PDF properties
# Used in place of func=torch.sigmoid
def exp_cdf(value):
    return (-(-value).exp()).exp()

def erf_cdf(value):
    return (torch.erf(value) + 1) / 2

def tanh_cdf(value):
    return (torch.tanh(value) + 1) / 2

# Example Wrappers for Torch Distribtuions
# However, it is better to just do
# func = torch.distributions.Normal(0, scale).cdf
# Might be useful if you want a learnable scale
def normal_cdf(value, scale=1.):
    return torch.distributions.Normal(0, scale).cdf(value)

def laplace_cdf(value, scale=1.):
    return torch.distributions.Laplace(0, scale).cdf(value)

def cauchy_cdf(value, scale=1.):
    return torch.distributions.Cauchy(0, scale).cdf(value)


# Wrapped Loss to avoid Softmax Accidentally
class OrdinalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, x, y):
        # x -> logits, size: (B, C)
        # y -> Labels, list (B) (like cross entropy)
        return self.loss(x, y)



# EOF
