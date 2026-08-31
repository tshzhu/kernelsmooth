import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from ..tools import (
    _gaussian, _pdist_sq,
    bw_scott, bw_silverman, optimize_bw,
)

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
EPS = 1e-10


class KernelDensity:
    # Gaussian Kernel Density Estimation

    __slots__ = (
        'bandwidth', 'method', 'scale', 'X', 'normalizer', # Basic attributes
        'diag_cov', 'L', # For whitening
        'neg2_Xt_scaled', 'X_scaled_sq', 'inv_bw', 'inv_bw_sq' # Cached for prediction
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
        self.bandwidth = None
        self.method = None # Only available for adaptive methods
        self.scale = 1.0 # Only available for rule-of-thumb methods

        try:
            match bandwidth:
                case str(method):
                    self.method = method
                case [str(method), scale]:
                    self.method = method
                    self.scale = np.asarray(scale, dtype=np.float64).ravel().squeeze()
                case _: # Fixed bandwidth (unknown dimension)
                    # self.method = None
                    self.bandwidth = np.asarray(bandwidth, dtype=np.float64).ravel().squeeze()
        except (ValueError, TypeError):
            raise ValueError(f"Invalid bandwidth format: {bandwidth}")

    def fit(self, X):
        # X shape: (n, d)
        self._whiten_fit(X) # compute self.X and self.L

        # bandwidth shape: (d,)
        self.bandwidth = self.get_bandwidth(self.X, self.method, self.scale)

        # Cache for prediction
        self._compute_cache()

        return self

    def get_bandwidth(self, X, method, scale=1.0):
        match method:
            # Fixed bandwidth
            case None:
                bw = self.bandwidth # ndarray
                dim = X.shape[1]

                if bw.ndim == 0:
                    return np.full(dim, bw)
                elif bw.shape[0] != dim:
                    raise ValueError(f"Bandwidth dimension mismatch. Expected {dim}, got {bw.shape[0]}")

                return bw

            # Rule-of-thumb
            case 'scott':
                return bw_scott(X, scale)
            case 'silverman':
                return bw_silverman(X, scale)

            # Cross-validation
            case 'mlcv':
                return bw_mlcv(X)
            case _:
                raise NotImplementedError(f"Unsupported bandwidth method: {method}")

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

        # X shape: (m, d)
        X = self._whiten_transform(X)
        X_scaled = X * self.inv_bw

        # dists_sq shape: (m, n)
        dists_sq = self._scaled_cdist_sq(X_scaled) # O(m*n)

        # K shape: (m, n)
        K = _gaussian(dists_sq) # O(m*n)

        # dens shape: (m,)
        dens = np.einsum('ij->i', K) # O(m*n)

        output = [dens * self.normalizer if normalize else dens,]

        if not return_grad:
            return output[0]

        # dens_grad shape: (m, d)
        # dens_grad = (K @ self.X - dens[:, None] * X) * self.inv_bw_sq
        dens_grad = K @ self.X
        X *= dens[:, None]
        dens_grad -= X
        dens_grad *= self.inv_bw_sq

        if self.diag_cov:
            dens_grad /= self.L
        else:
            solve_triangular(self.L, dens_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

        output.append(dens_grad * self.normalizer if normalize else dens_grad)

        return output

    def _whiten_fit(self, X):
        X = np.atleast_2d(X).astype(np.float64)
        n, dim = X.shape

        if self.diag_cov:
            # std = L
            # X_whiten = X / L
            self.L = np.std(X, axis=0, ddof=1)
            self.X = X / self.L
        else:
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            cov = X_centered.T @ X_centered / (n - 1)

            cov.flat[::dim + 1] += EPS  # Avoid singular covariance

            # cov = L @ L^T
            # X_whiten = X @ L^(-T)
            # Use `overwrite_a=True` to inplace modify `cov`
            self.L = cholesky(cov.T, lower=True, check_finite=False, overwrite_a=True)
            self.X = solve_triangular(self.L, X.T, lower=True, check_finite=False).T

    def _whiten_transform(self, X):
        X = np.atleast_2d(X).astype(np.float64)

        if self.diag_cov:
            return X / self.L
        else:
            return solve_triangular(self.L, X.T, lower=True, check_finite=False).T

    def _scaled_cdist_sq(self, X):
        # XX shape: (m,)
        XX = np.einsum('ij,ij->i', X, X)

        # dists_sq shape: (m, n)
        # dists_sq = -2.0 * X @ self.X_scaled.T + XX[:, None] + self.X_scaled_sq
        dists_sq = X @ self.neg2_Xt_scaled
        dists_sq += XX[:, None]
        dists_sq += self.X_scaled_sq

        np.maximum(dists_sq, 0.0, out=dists_sq)

        return dists_sq

    def _compute_cache(self):
        n, dim = self.X.shape

        self.inv_bw = (1.0 / self.bandwidth)[None, :]
        self.inv_bw_sq = self.inv_bw ** 2

        X_scaled = self.X * self.inv_bw
        self.X_scaled_sq = np.einsum('ij,ij->i', X_scaled, X_scaled)[None, :]
        self.neg2_Xt_scaled = (-2.0 * X_scaled).T

        # Normalization factor for PDF
        # 1/n * (2 * pi)^(-d/2) * det(H)^(-1/2)
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
        dists_sq = _pdist_sq(X / np.exp(log_h))
        exponents = -0.5 * dists_sq
        np.fill_diagonal(exponents, -np.inf)

        log_dens = logsumexp(exponents, axis=1)

        # negative log-likelihood
        return -np.mean(log_dens) + log_h.sum()

    return optimize_bw(loss, bw_silverman(X))
