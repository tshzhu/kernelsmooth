import numpy as np
from scipy.linalg import solve_triangular

from .kernel_density import KernelDensity
from ..tools import (
    _gaussian, _pdist_sq,
    bw_silverman, optimize_bw,
)


class KernelRegression(KernelDensity):
    # Gaussian Nadaraya-Watson Kernel Regression

    __slots__ = ('Y', 'X_Y')

    def __init__(self, bandwidth='lscv', diag_cov=False):
        """
        Initializes the Kernel Regression model.

        Args:
            bandwidth (float, array_like, str, or tuple, optional): The bandwidth selection method or value.
                - If float or array_like: Fixed bandwidth value(s).
                - If str: Name of the adaptive method (e.g., 'scott', 'silverman', 'lscv').
                - If tuple: A pair of (method_name, scale_factor), e.g., ('scott', 0.5).
            diag_cov (bool, optional): Whether to use a diagonal covariance approximation for whitening.
                - If True: Performs component-wise standardization (diagonal whitening).
                  The kernel becomes: k(x, x') = exp(-0.5 * ||(x - x') / (std * h)||^2).
                - If False: Performs full whitening using Cholesky decomposition of the covariance matrix.
                  The kernel uses the Mahalanobis distance structure.
        """

        super().__init__(bandwidth=bandwidth, diag_cov=diag_cov)

    def fit(self, X, Y):
        # X shape: (n, d)
        self._whiten_fit(X) # compute self.X and self.L

        # Y shape: (n,)
        self.Y = np.ravel(Y).astype(np.float64)

        # bandwidth shape: (d,)
        self.bandwidth = self.get_bandwidth(self.X, self.Y, self.method, self.scale)

        # Cache for prediction
        self._compute_cache()
        self.X_Y = self.X * self.Y[:, None] # O(n*d)

        return self

    def get_bandwidth(self, X, Y, method, scale=1.0):
        match method:
            case 'lscv':
                return bw_lscv(X, Y)
            case 'gcv':
                return bw_gcv(X, Y)
            case 'aicc':
                return bw_aicc(X, Y)
            case 'recp':
                return bw_recp(X, Y)
            case _:
                # Use bandwidth selection methods for kernel density
                return super().get_bandwidth(X, method, scale)

    def predict(self, X, return_dens=False, normalize=False, return_grad=False):
        """
        Predicts the regression mean and optionally the kernel density and their gradients.

        Args:
            X (np.ndarray): Query points with shape (m, d).
            return_dens (bool, optional): If True, returns the density estimate (and its
                gradient if `return_grad` is True) in addition to the mean. Defaults to False.
            normalize (bool, optional): If True, normalizes the density estimate and its
                gradient by the standard Gaussian kernel normalization factor.
                Only used if `return_dens` is True. Defaults to False.
            return_grad (bool, optional): If True, returns the gradients of the mean
                (and density if `return_dens` is True) with respect to the input `X`.
                Defaults to False.

        Returns:
            np.ndarray | list:
                - If `return_dens` and `return_grad` are False:
                    Returns **mean** (np.ndarray): The regression mean values, shape (m,).

                - Otherwise, returns a list containing the requested outputs in the following order:
                    1. **mean** (np.ndarray): Shape (m,).
                    2. **mean_grad** (np.ndarray, optional): Gradient of mean w.r.t X, shape (m, d).
                       (Included if `return_grad` is True).
                    3. **dens** (np.ndarray, optional): Density values, shape (m,).
                       (Included if `return_dens` is True).
                    4. **dens_grad** (np.ndarray, optional): Gradient of density w.r.t X, shape (m, d).
                       (Included if both `return_dens` and `return_grad` are True).

        Examples:
            >>> mean = model.predict(X)
            >>> mean, mean_grad = model.predict(X, return_grad=True)
            >>> mean, dens = model.predict(X, return_dens=True)
            >>> mean, mean_grad, dens, dens_grad = model.predict(X, return_dens=True, return_grad=True)
        """

        # X shape: (m, d)
        X = self._whiten_transform(X)
        X_scaled = X * self.inv_bw

        # dists_sq shape: (m, n)
        dists_sq = self._scaled_cdist_sq(X_scaled) # O(m*n)
        d_min_sq = np.min(dists_sq, axis=1) # O(m*n)

        # Kernel weights with shift for numerical stability
        # K_shifted shape: (m, n)
        dists_sq -= d_min_sq[:, None]
        K_shifted = _gaussian(dists_sq) # O(m*n)

        # dens_shifted shape: (m,)
        dens_shifted = np.einsum('ij->i', K_shifted) # O(m*n)

        # mean shape: (m,)
        mean = (K_shifted @ self.Y) / dens_shifted # O(m*n)

        if not (return_dens or return_grad):
            return mean

        output = [mean,]

        if return_grad:
            # K_X shape: (m, d)
            K_X = K_shifted @ self.X

            # mean_grad shape: (m, d)
            # K_XY = (K_shifted * self.Y) @ self.X = K_shifted @ (self.X * self.Y[:, None])
            # numerator = K_XY - mean[:, None] * K_X
            # mean_grad = numerator * (self.inv_bw_sq / dens_shifted[:, None])
            mean_grad = K_shifted @ self.X_Y
            mean_grad -= mean[:, None] * K_X
            mean_grad *= self.inv_bw_sq / dens_shifted[:, None]

            if self.diag_cov:
                mean_grad /= self.L
            else:
                solve_triangular(self.L, mean_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

            output.append(mean_grad)

        if return_dens:
            # shift_factor shape: (m,)
            shift_factor = _gaussian(d_min_sq) # O(m)

            # dens shape: (m,)
            dens = dens_shifted * shift_factor # O(m)

            output.append(dens * self.normalizer if normalize else dens)

            if return_grad:
                # dens_grad shape: (m, d)
                # dens_grad_shifted = (K_X - dens_shifted[:, None] * X) * self.inv_bw_sq
                # dens_grad = dens_grad_shifted * shift_factor[:, None]
                dens_grad = K_X
                X *= dens_shifted[:, None]
                dens_grad -= X
                dens_grad *= self.inv_bw_sq
                dens_grad *= shift_factor[:, None]

                if self.diag_cov:
                    dens_grad /= self.L
                else:
                    solve_triangular(self.L, dens_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

                output.append(dens_grad * self.normalizer if normalize else dens_grad)

        return output


# -----------------------------------------------------------
# Bandwidth Selection for KR
# -----------------------------------------------------------

def _get_hat_stats(X, Y, h):
    dists_sq = _pdist_sq(X / h)

    # K shape: (n, n)
    K = _gaussian(dists_sq)

    # Hat Statistics
    H_diag = 1.0 / np.einsum('ij->i', K) # inv_dens
    Y_hat = (K @ Y) * H_diag # mean

    return Y_hat, H_diag

def bw_lscv(X, Y):
    """
    Least-Squares Cross Validation
    """
    def loss(log_h):
        Y_hat, H_diag = _get_hat_stats(X, Y, np.exp(log_h))

        denominator = np.maximum(1.0 - H_diag, 1e-8)

        residuals = (Y - Y_hat) / denominator
        return np.mean(residuals ** 2)

    return optimize_bw(loss, bw_silverman(X))

def bw_gcv(X, Y):
    """
    Generalized Cross Validation
    GCV: MSE / (1 - tr(H)/n)^2
    """
    n = len(Y)

    def loss(log_h):
        Y_hat, H_diag = _get_hat_stats(X, Y, np.exp(log_h))
        tr_H = H_diag.sum()

        denominator = np.maximum(1.0 - tr_H / n, 1e-6)

        mse = np.mean((Y - Y_hat) ** 2)
        return mse / (denominator ** 2)

    return optimize_bw(loss, bw_silverman(X))

def bw_aicc(X, Y):
    """
    Corrected AIC
    Better for small samples than standard AIC.
    AICc: log(MSE) + (1 + tr(H)/n) / (1 - (tr(H)+2)/n)
    """
    n = len(Y)

    def loss(log_h):
        Y_hat, H_diag = _get_hat_stats(X, Y, np.exp(log_h))
        tr_H = H_diag.sum()

        numerator = 1.0 + tr_H / n
        denominator = np.maximum(1.0 - (tr_H + 2.0) / n, 1e-6)

        mse = np.maximum(np.mean((Y - Y_hat) ** 2), 1e-20)
        return np.log(mse) + (numerator / denominator)

    return optimize_bw(loss, bw_silverman(X))

def bw_recp(X, Y):
    """
    Risk Estimation using Classical Pilots
    """
    n = len(Y)

    # Pilot Estimation
    h_pilot = bw_lscv(X, Y)

    Y_hat_pilot, H_diag_pilot = _get_hat_stats(X, Y, h_pilot)
    tr_H_pilot = H_diag_pilot.sum()

    mse_pilot = np.mean((Y - Y_hat_pilot) ** 2)
    var_pilot = mse_pilot / (1.0 - tr_H_pilot / n)

    def loss(log_h):
        dists_sq = _pdist_sq(X / np.exp(log_h))
        K = _gaussian(dists_sq)

        # Hat Statistics
        H_diag = 1.0 / np.einsum('ij->i', K)
        Y_hat = (K @ Y_hat_pilot) * H_diag

        sum_K_sq = np.einsum('ij,ij->i', K, K)
        tr_HHt = np.einsum('i,i,i->', sum_K_sq, H_diag, H_diag)
        bias_sq = np.sum((Y_hat_pilot - Y_hat) ** 2)
        variance = var_pilot * tr_HHt

        risk = (bias_sq + variance) / n
        return risk

    return optimize_bw(loss, bw_silverman(X))
