from dpm.emd import emd
from dpm.distributions import Normal
import numpy as np


def test_emd():
    P_r = np.array([12,7,4,1,19,14,9,6,3,2])
    P_t = np.array([1,5,11,17,13,9,6,4,3,2])

    P_r = P_r / np.sum(P_r)
    P_t = P_t / np.sum(P_t)

    emd_primal, gamma_primal = emd(P_r, P_t)

    assert gamma_primal.shape == (len(P_r), len(P_t))

    # valid marginal
    assert (gamma_primal.sum() - 1.0 < 1e-5)
    # assert constraints followed
    assert (gamma_primal.sum(0) - P_t < 1e-5).all()
    assert (gamma_primal.sum(1) - P_r < 1e-5).all()

    emd_dual, (f, g) = emd(P_r, P_t, dual=True)

    assert (emd_dual - emd_primal < 1e-5)

    assert f.shape == (len(P_r),)
    assert g.shape == (len(P_t),)

    assert (np.abs(f.sum()) - 1 < 1e-5)
    assert (np.abs(g.sum()) - 1 < 1e-5)

    assert ((f + g).sum() - 0 < 1e-5)


def test_emd_distribution():
    p_model = Normal(0., 1.)
    q_model = Normal(-4., 3.)

    emd_primal, gamma_primal = emd(p_model, q_model, batch_size=1024, n_bins=20)
    emd_dual, (f, g) = emd(p_model, q_model, dual=True, batch_size=1024, n_bins=20)

    assert gamma_primal.sum() - 1. < 1e-2
