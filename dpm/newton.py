import torch
import numpy as np

# FIRST ORDER

# (loss, input)
def gradient(ys, xs):
    if len(ys.shape) == 0:
        return grad(ys, xs)
    return jacobian(ys, xs)

# (loss scalar, input)
def grad(y, xs):
    return torch.autograd.grad(y, xs, create_graph=True,
                               retain_graph=True, allow_unused=True)[0]

# (loss vector, input vector)
def jacobian(ys, xs):
    return torch.stack([grad(yi, xs) for yi in ys])


# SECOND ORDER

# (loss, input)
def hessian(ys, xs):
    if len(ys.shape) == 0:
        return hessian_1d(ys, xs)
    return hessian_2d(ys, xs)

# loss scalar, input
def hessian_1d(y, xs):
    dys = gradient(y, xs)
    flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
    H = torch.stack([torch.cat([Hij.reshape(-1) for Hij in gradient(dyi, xs)])
                     for dyi in flat_dy]).reshape((*xs.shape, *xs.shape))
    return H

# loss vector, input
def hessian_2d(ys, xs, return_grad=False):
    return torch.stack([hessian_1d(y, xs) for y in ys])


# NEWTON STEP

# compute newton step: -H^-1 * g
# scalar y, any input xs
def newton_step(y, xs, use_pinv=False):
    g = gradient(y, xs)
    H = hessian(y, xs)

    rows = np.prod(H.shape[:len(H.shape)//2])
    H = H.reshape((rows, rows))
    g = g.reshape(-1)

    if use_pinv:
        Hinv = torch.pinverse(H)
    else:
        Hinv = torch.inverse(H)
    ret = -torch.mv(Hinv, g)
    return ret.reshape(xs.shape)
