import torch
from .constants import eps
import numpy as np

def sqrtx2p1(x):
    return x.abs() * (1 + x.pow(-2)).sqrt()

def softplus_inverse(value, threshold=20):
    inv = torch.log((value.exp() - 1.0))
    inv[value > threshold] = value[value > threshold]
    return inv

def logit(x):
    return x.log() - (-x).log1p()

def log(x):
    return (x + eps).log()

# Kronecker
def kron(A, B):
    return torch.einsum('ij,kl->ikjl', [A, B]).view(A.size(0) * B.size(0),
                                                    A.size(1) * B.size(1))

def vec(A):
    return torch.einsum("ij->ji", [A]).contiguous().view(-1, 1)

# https://rockt.github.io/2018/04/30/einsum#fn.10

def transpose(A):
    return torch.einsum('ij->ji', [A])

def sum(A):
    return torch.einsum('ij->', [A.view(1, -1)])

def col_sum(A):
    return torch.einsum('ij->j', [A])

def row_sum(A):
    return torch.einsum('ij->i', [A]).view(-1, 1)

def mv(mat, vec):
    return torch.einsum('ik,k->i', [mat, vec])

def mm(A, B):
    return torch.einsum('ik,kj->ij', [A, B])

def dot(a, b):
    return torch.einsum('ij,ij->', [a.view(-1, 1), b.view(-1, 1)])

def hadamard(A, B):
    return torch.einsum('ij,ij->ij', [A, B])

def outer_product(a, b):
    return torch.einsum('i,j->ij', [a, b])

def bmm(A, B):
    return torch.einsum('ijk,ikl->ijl', [A, B])

# only square
def diag(A):
    return torch.einsum('ii->i', A)

def batch_diag(A):
    return torch.einsum('...ii->...i', A)

def bilinear(l, A, r):
    return torch.einsum('bn,anm,bm->ba', l, A, r)


def proj(v, u):
    return (v.dot(u) / (u.dot(u))) * u


# generic inverse
def inverse(X):
    assert len(X.shape) % 2 == 0
    assert X.shape[:len(X.shape)//2] == X.shape[len(X.shape)//2:]
    original = X.shape
    rows = np.prod(X.shape[:len(X.shape)//2])
    return torch.inverse(X.reshape(rows, rows)).reshape(original)

def pinverse(X):
    assert len(X.shape) % 2 == 0
    assert X.shape[:len(X.shape)//2] == X.shape[len(X.shape)//2:]
    original = X.shape
    rows = np.prod(X.shape[:len(X.shape)//2])
    return torch.pinverse(X.reshape(rows, rows)).reshape(original)


def cov(X, Y=None):
    if Y is None:
        Y = X
    Xm = X - X.mean(dim=0).expand_as(X)
    Ym = Y - Y.mean(dim=0).expand_as(Y)
    Xm, Ym = Xm.t(), Ym.t()
    C = Xm.mm(Ym.t()) / X.size(0)
    return C


def corr(X, Y=None):
    if Y is None:
        Y = X
    C = cov(X, Y)
    std_x = X.std(dim=0)
    std_y = Y.std(dim=0)
    C = C / std_x.expand_as(C)
    C = C / std_y.expand_as(C).t()
    return C


# Testing Difference
def to_hist(X, bins=50, min=0, max=0.):
    return torch.histc(X, bins=bins, min=min, max=max)


def bincount(samples, bins, num_bins):
    idxs, counts = np.unique(np.digitize(samples, bins) - 1, return_counts=True)
    bincounts = np.zeros(num_bins + 1)
    bincounts[idxs] = counts
    return bincounts


def model_to_bins(p_model, q_model, batch_size=64, n_bins=10):
    p_samples = p_model.sample(batch_size).detach().numpy()
    q_samples = q_model.sample(batch_size).detach().numpy()
    total_samples = np.concatenate((p_samples, q_samples), axis=0)
    _, bins = np.histogram(total_samples, bins=n_bins)
    p_hist = bincount(p_samples, bins, n_bins)
    q_hist = bincount(q_samples, bins, n_bins)
    return p_hist / np.sum(p_hist), q_hist / np.sum(q_hist)


def percentile_rank(samples):
    return samples.view(-1).argsort().argsort().float() / samples.size(0)


def kl(h1, h2):
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    return (h1 * log(h1) - h1 * log(h2)).sum()


def integrate(model, rng=(-10, 10), n_points=10000):
    x = np.linspace(rng[0], rng[1], n_points)
    probs = model.log_prob(torch.tensor(x).view(-1, 1).float()).exp().detach().numpy()
    probs = np.nan_to_num(probs)
    return np.trapz(probs, x)
