import torch
from .constants import eps

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
