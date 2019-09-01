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
