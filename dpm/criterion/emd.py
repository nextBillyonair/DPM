import numpy as np
import torch
from scipy.optimize import linprog
from dpm.distributions import Distribution
from dpm.utils import bincount, model_to_bins

# https://vincentherrmann.github.io/blog/wasserstein/

# Make cost matrix
def make_distance_matrix(p_len, q_len):
    return np.abs(np.arange(p_len).reshape(-1, 1) -
                  np.arange(q_len).reshape(1, -1))


def make_constraint_matrix(p_len, q_len):
    n = p_len * q_len
    m = p_len + q_len

    A = np.zeros((n, m))

    for i in range(p_len):
        for j in range(q_len):
            r_id = i*q_len + j
            A[r_id, i] = 1

    for i in range(p_len):
        for j in range(q_len):
            r_id = q_len * i + j
            c_id = p_len + j
            A[r_id, c_id] = 1

    return A.T


# P, Q must be discrete histograms
# rows -> p_len, cols ->q_len
def emd(p_model, q_model, batch_size=64, dual=False, n_bins=10):

    if isinstance(p_model, Distribution) and isinstance(q_model, Distribution):
        p_model, q_model = model_to_bins(p_model, q_model, batch_size, n_bins)

    p_len = len(p_model)
    q_len = len(q_model)

    D = make_distance_matrix(p_len, q_len)
    A = make_constraint_matrix(p_len, q_len)
    b = np.concatenate((p_model, q_model), axis=0)
    c = D.flatten()

    if dual:
        # linprog() can only minimize the cost, because of that
        # we optimize the negative of the objective. Also, we are
        # not constrained to nonnegative values.
        opt_res = linprog(-b, A.T, c, bounds=(None, None))
        emd = -opt_res.fun
        f = opt_res.x[0:p_len]
        g = opt_res.x[p_len:]
        return emd, (f, g)

    # primal
    opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
    emd = opt_res.fun
    gamma = opt_res.x.reshape((p_len, q_len))
    return emd, gamma


# EOF
