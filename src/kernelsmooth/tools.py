"""Numerical helpers shared by the kernel-smoothing estimators.

The public estimators whiten their inputs before calling these functions, so
the bandwidth rules operate in a standardized coordinate system.  Distance
helpers deliberately clip small negative round-off errors before a square root
or Gaussian evaluation is applied.
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# Bandwidth optimization is restricted to a factor of five around its initial
# value.  This keeps the inexpensive numerical search from drifting toward
# degenerate near-zero or excessively large bandwidths.
LOG_THRES = np.log(5.0)
EPS = 1e-10


def _cdist_sq(X, Y):
    """Return pairwise squared Euclidean distances between two point sets."""

    XX = np.einsum('ij,ij->i', X, X)
    YY = np.einsum('ij,ij->i', Y, Y)
    dists_sq = -2.0 * X @ Y.T + XX[:, None] + YY[None, :]
    # Matrix multiplication can produce tiny negative values for equal points.
    return np.maximum(dists_sq, 0.0)

def _pdist_sq(X):
    """Return the full matrix of squared distances within one point set."""

    XX = np.einsum('ij,ij->i', X, X)
    dists_sq = -2.0 * X @ X.T + XX[:, None] + XX[None, :]
    return np.maximum(dists_sq, 0.0)

def _gaussian(dists_sq):
    """Convert squared distances to Gaussian weights in place."""

    # Reusing the distance array avoids allocating another query-by-training
    # matrix in code paths that already materialize pairwise distances.
    dists_sq *= -0.5
    np.exp(dists_sq, out=dists_sq)

    return dists_sq


def bw_scott(X, scale=1.0):
    """Return Scott's rule-of-thumb bandwidth in whitened coordinates."""

    n, dim = X.shape
    factor = n ** (-1.0 / (dim + 4.0))

    # Feature scales are already represented by the whitening transform.
    bw = scale * 1.06 * factor
    return np.full(dim, bw)

def bw_silverman(X, scale=1.0):
    """Return a robust Silverman bandwidth using per-feature IQR estimates."""

    n, dim = X.shape
    factor = (n * (dim + 2.0) / 4.0) ** (-1.0 / (dim + 4.0))

    # np.partition obtains the required quartiles without fully sorting every
    # feature.  Dividing by 1.3489795 converts a normal-distribution IQR to a
    # standard-deviation estimate.
    k25 = int(0.25 * n)
    k75 = int(0.75 * n)
    part = np.partition(X, [k25, k75], axis=0)
    iqr_value = part[k75] - part[k25]
    iqr_value /= 1.3489795
    iqr_value = np.maximum(EPS, iqr_value)

    # Whitening makes one the natural upper bound for the robust scale.
    A = np.minimum(1.0, iqr_value)

    bw = scale * A * factor
    return bw

def bw_median(X, scale=1.0):
    """Return an isotropic bandwidth from the median pairwise distance."""

    n, dim = X.shape

    dists_sq = _pdist_sq(X)
    # Exclude the zero diagonal and count each unordered pair once.
    inds = np.triu_indices(n, k=1)
    pairwise_dists = np.sqrt(dists_sq[inds])

    med_dist = np.median(pairwise_dists)
    med_dist = np.maximum(med_dist, 1e-8)

    bw = scale * med_dist
    return np.full(dim, bw)


def optimize_bw(loss, h0):
    """Minimize a bandwidth criterion in log space around ``h0``.

    Optimizing ``log(h)`` enforces positive bandwidths while allowing one
    independent value per feature.  Numerical gradients keep the callback
    interface simple for the several cross-validation criteria in this package.
    """

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
        maxiter=10, # Keep automatic selection bounded and inexpensive.
    )
    return np.exp(log_h)
