"""Cython-accelerated Gaussian kernel density estimation."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from ._kernel_density import _get_kde, _get_kde_grad, _get_kde_loo
from .tools import bw_scott, bw_silverman, bw_median, optimize_bw, EPS

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class KernelDensity:
    """Gaussian KDE with covariance whitening and analytic gradients."""

    __slots__ = (
        'bandwidth', 'method', 'scale', 'X', 'normalizer', # Fitted model state
        'diag_cov', 'L', # Whitening mode and linear transform
        'X_scaled', 'inv_bw', # Prediction-time cache
    )

    def __init__(self, bandwidth='mlcv', diag_cov=False):
        """
        Initializes the Kernel Density Estimator.

        Args:
            bandwidth (float, array_like, str, or tuple, optional): The bandwidth selection method or value.
                - If float or array_like: Fixed bandwidth value(s).
                - If str: Name of the adaptive method (e.g., 'scott', 'silverman', 'mlcv').
                - If tuple: A pair of (method_name, scale_factor), e.g., ('scott', 0.5).
            diag_cov (bool, optional): Whether to use a diagonal covariance approximation for whitening.
                - If True: Performs component-wise standardization (diagonal whitening).
                  The kernel becomes: k(x, x') = exp(-0.5 * ||(x - x') / (std * h)||^2).
                - If False: Performs full whitening using Cholesky decomposition of the covariance matrix.
                  The kernel uses the Mahalanobis distance structure.
        """

        self.set_bandwidth(bandwidth)
        self.diag_cov = diag_cov

    def set_bandwidth(self, bandwidth):
        """Configure a fixed bandwidth or an automatic selection method.

        Parsing uses temporary variables so a rejected update does not corrupt
        the configuration of an existing estimator.
        """

        fixed_bandwidth = None
        method = None
        scale = 1.0

        try:
            match bandwidth:
                case str(method):
                    pass
                case [str(method), scale]:
                    scale = np.asarray(scale, dtype=np.float64).ravel().squeeze()
                case _: # Fixed bandwidth (unknown dimension)
                    fixed_bandwidth = np.asarray(bandwidth, dtype=np.float64).ravel().squeeze()
        except (ValueError, TypeError):
            raise ValueError(f"Invalid bandwidth format: {bandwidth}")

        if fixed_bandwidth is not None:
            if not np.all(np.isfinite(fixed_bandwidth)):
                raise ValueError("Fixed bandwidth values must be finite")
            if not np.all(fixed_bandwidth > 0.0):
                raise ValueError("Fixed bandwidth values must be strictly positive")

        self.bandwidth = fixed_bandwidth
        self.method = method
        self.scale = scale

    def fit(self, X):
        """Fit the whitening transform, select bandwidths, and cache data."""

        # X and the stored training matrix both have shape (n, d).
        self.X = self.whiten_fit_transform(X) # Also fits self.L.

        # A scalar fixed bandwidth is expanded here once the dimension is known.
        self.bandwidth = self.get_bandwidth(self.X, self.method, self.scale)

        # Prediction repeatedly reuses bandwidth-scaled training coordinates.
        self._compute_cache()

        return self

    def get_bandwidth(self, X, method, scale=1.0):
        """Resolve a bandwidth specification for data with shape ``(n, d)``."""

        match method:
            case 'scott': return bw_scott(X, scale)
            case 'silverman': return bw_silverman(X, scale)
            case 'median': return bw_median(X, scale)
            case 'mlcv': return bw_mlcv(X)

            case None: # Use the fixed bandwidth configured by set_bandwidth.
                bw = self.bandwidth
                dim = X.shape[1]

                if bw.ndim == 0:
                    return np.full(dim, bw)
                elif bw.shape[0] != dim:
                    raise ValueError(f"Bandwidth dimension mismatch. Expected {dim}, got {bw.shape[0]}")
                return bw

            case _: raise NotImplementedError(f"Unsupported bandwidth method: {method}")

    def pdf(self, X, normalize=False, return_grad=False):
        """
        Computes the kernel density estimates and optionally their gradients.

        Args:
            X (np.ndarray): Query points with shape (m, d).
            normalize (bool, optional): If True, multiplies the density and gradient by
                the standard Gaussian normalization factor (1 / (n * h^d * (2pi)^(d/2))).
                Defaults to False (returns unnormalized sum of kernels).
            return_grad (bool, optional): If True, returns the gradient of the density
                with respect to the input X. Defaults to False.

        Returns:
            np.ndarray | list:
                - If `return_grad` is False:
                    Returns **dens** (np.ndarray): The density values at X, shape (m,).

                - If `return_grad` is True:
                    Returns a list ``[dens, grad]``:
                    1. **dens** (np.ndarray): Density values, shape (m,).
                    2. **grad** (np.ndarray): Gradient of density w.r.t X, shape (m, d).

        Examples:
            >>> dens = model.predict(X)
            >>> dens, dens_grad = model.predict(X, return_grad=True)
        """

        # Transform once in Python; the O(m*n*d) kernel accumulation runs in
        # Cython without materializing the complete pairwise kernel matrix.
        X = self.whiten_transform(X)
        X_scaled = X * self.inv_bw

        if not return_grad:
            # dens shape: (m,)
            dens = _get_kde(X_scaled, self.X_scaled)

            if normalize:
                dens *= self.normalizer
            return dens

        # The Cython routine returns sum(K) and sum(K * x_i / h).  Combining
        # those moments produces d sum(K) / d x in whitened coordinates.
        dens, dens_grad = _get_kde_grad(X_scaled, self.X_scaled)
        X_scaled *= dens[:, None]
        dens_grad -= X_scaled
        dens_grad *= self.inv_bw

        # Apply the chain rule through the whitening map so public gradients
        # are always expressed in the original input coordinates.
        if self.diag_cov:
            dens_grad /= self.L
        else:
            solve_triangular(self.L, dens_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

        if normalize:
            dens *= self.normalizer
            dens_grad *= self.normalizer
        return dens, dens_grad

    def whiten_fit_transform(self, X):
        """Fit the covariance transform and return whitened coordinates.

        Only the linear transform is applied to coordinates.  This is
        sufficient because Gaussian kernels depend on pairwise differences,
        and it keeps ``whiten_inverse_transform`` meaningful for offsets.
        """

        X = np.atleast_2d(X).astype(np.float64)
        n, dim = X.shape

        if n < 2:
            raise ValueError("X must contain at least 2 samples")

        if self.diag_cov:
            # Diagonal whitening: X_white = X / per-feature sample std.
            self.L = np.std(X, axis=0, ddof=1)
            return X / self.L
        else:
            # Centering is used to estimate covariance, not stored coordinates;
            # the common translation cancels from every kernel difference.
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            cov = X_centered.T @ X_centered / (n - 1)

            cov.flat[::dim + 1] += EPS  # Avoid singular covariance

            # cov = L @ L.T and X_white = solve(L, X.T).T.
            # ``overwrite_a`` lets SciPy reuse the temporary covariance buffer.
            self.L = cholesky(cov.T, lower=True, check_finite=False, overwrite_a=True)
            return solve_triangular(self.L, X.T, lower=True, check_finite=False).T

    def whiten_transform(self, X):
        """Apply the fitted linear whitening transform to query points."""

        X = np.atleast_2d(X).astype(np.float64)

        if self.diag_cov:
            return X / self.L
        else:
            return solve_triangular(self.L, X.T, lower=True, check_finite=False).T

    def whiten_inverse_transform(self, Z):
        """Map whitened offsets back to offsets in the original coordinates."""

        Z = np.atleast_2d(Z).astype(np.float64)

        if self.diag_cov:
            return Z * self.L
        else:
            return Z @ self.L.T

    def _compute_cache(self):
        """Cache scaled training coordinates and the normalized KDE factor."""

        n, dim = self.X.shape

        self.inv_bw = (1.0 / self.bandwidth)[None, :]
        self.X_scaled = self.X * self.inv_bw

        # 1/n * (2*pi)^(-d/2) * prod(1/h) / det(L) converts the
        # unnormalized Gaussian sum back to density in original coordinates.
        if self.diag_cov:
            det_L = np.prod(self.L)
        else:
            det_L = np.prod(np.diag(self.L))

        self.normalizer = (INV_SQRT_2PI ** dim) * self.inv_bw.prod() / det_L / n


# -----------------------------------------------------------
# Bandwidth Selection for KDE
# -----------------------------------------------------------

def bw_mlcv(X):
    """
    Maximum-Likelihood Cross Validation
    """
    def loss(log_h):
        # Optimize in log space; _get_kde_loo omits each point's self-kernel.
        h = np.exp(log_h)
        dens = _get_kde_loo(X, h)

        log_dens = np.log(dens)

        # negative log-likelihood
        return -np.mean(log_dens) + log_h.sum()

    return optimize_bw(loss, bw_silverman(X))
