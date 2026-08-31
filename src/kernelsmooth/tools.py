import numpy as np
from scipy.optimize import fmin_l_bfgs_b

LOG_THRES = np.log(5.0)
EPS = 1e-10


def _cdist_sq(X, Y):
    XX = np.einsum('ij,ij->i', X, X)
    YY = np.einsum('ij,ij->i', Y, Y)
    dists_sq = -2.0 * X @ Y.T + XX[:, None] + YY[None, :]
    return np.maximum(dists_sq, 0.0)

def _pdist_sq(X):
    XX = np.einsum('ij,ij->i', X, X)
    dists_sq = -2.0 * X @ X.T + XX[:, None] + XX[None, :]
    return np.maximum(dists_sq, 0.0)

def _gaussian(dists_sq):
    # Inplace computation of Gaussian kernel
    # K = np.exp(-0.5 * dists_sq)
    dists_sq *= -0.5
    np.exp(dists_sq, out=dists_sq)

    return dists_sq


def bw_scott(X, scale=1.0):
    n, dim = X.shape
    factor = n ** (-1.0 / (dim + 4.0))

    # Note that std = 1.0 after whitening
    bw = scale * 1.06 * factor
    return np.full(dim, bw)

def bw_silverman(X, scale=1.0):
    n, dim = X.shape
    factor = (n * (dim + 2.0) / 4.0) ** (-1.0 / (dim + 4.0))

    k25 = int(0.25 * n)
    k75 = int(0.75 * n)
    part = np.partition(X, [k25, k75], axis=0)
    iqr_value = part[k75] - part[k25]
    iqr_value /= 1.3489795
    iqr_value = np.maximum(EPS, iqr_value)

    # Note that std = 1.0 after whitening
    A = np.minimum(1.0, iqr_value)

    bw = scale * A * factor
    return bw

def bw_median(X, scale=1.0):
    n, dim = X.shape

    dists_sq = _pdist_sq(X)
    inds = np.triu_indices(n, k=1)
    pairwise_dists = np.sqrt(dists_sq[inds])

    med_dist = np.median(pairwise_dists)
    med_dist = np.maximum(med_dist, 1e-8)

    bw = scale * med_dist
    return np.full(dim, bw)


def optimize_bw(loss, h0):
    log_h0 = np.log(h0)
    bounds = np.column_stack((
        log_h0 - LOG_THRES,
        log_h0 + LOG_THRES
    ))

    log_h, _, _ = fmin_l_bfgs_b(
        func=loss,
        x0=log_h0,
        bounds=bounds,
        approx_grad=True,
        maxiter=10, # Avoid overfitting
    )
    return np.exp(log_h)
