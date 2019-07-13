import numpy as np
from scipy.optimize import linprog

# https://vincentherrmann.github.io/blog/wasserstein/

# Make cost matrix
def make_distance_matrix(pr_len, pt_len):
    D = np.ndarray(shape=(pr_len, pt_len))

    for i in range(pr_len):
        for j in range(pt_len):
            D[i,j] = abs(range(pr_len)[i] - range(pt_len)[j])
    return D


def make_A_matrix(pr_len, pt_len):
    n = pr_len * pt_len
    m = pr_len + pt_len

    A = np.zeros((n, m))

    for i in range(pr_len):
        for j in range(pt_len):
            r_id = i*pt_len + j
            A[r_id, i] = 1

    for i in range(pr_len):
        for j in range(pt_len):
            r_id = pt_len * i + j
            c_id = pr_len + j
            A[r_id, c_id] = 1

    return A.T


# P, Q must be discrete histograms
# Pr = P     Ptheta = Q
# rows -> p_len, cols ->q_len
def emd(P_r, P_t, dual=False):
    pr_len = len(P_r)
    pt_len = len(P_t)

    D = make_distance_matrix(pr_len, pt_len)
    A = make_A_matrix(pr_len, pt_len)
    b = np.concatenate((P_r, P_t), axis=0)
    c = D.flatten()

    if dual:
        # linprog() can only minimize the cost, because of that
        # we optimize the negative of the objective. Also, we are
        # not constrained to nonnegative values.
        opt_res = linprog(-b, A.T, c, bounds=(None, None))
        emd = -opt_res.fun
        f = opt_res.x[0:pr_len]
        g = opt_res.x[pr_len:]
        print("dual EMD: ", emd)
        return emd, (f, g)

    # primal
    opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
    emd = opt_res.fun
    gamma = opt_res.x.reshape((pr_len, pt_len))
    print("EMD: ", emd, "\n")
    return emd, gamma











# EOF
