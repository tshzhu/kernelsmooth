import numpy as np
from scipy.linalg import solve_triangular

from .kernel_density import KernelDensity
from ._kernel_regression import _get_kr, _get_kr_grad, _get_kr_loo
from .tools import _gaussian, bw_silverman, optimize_bw


class KernelRegression(KernelDensity):
    # Gaussian Nadaraya-Watson Kernel Regression

    __slots__ = ('Y',)

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
        self.X = self.whiten_fit_transform(X) # fit self.X and self.L

        # Y shape: (n,)
        self.Y = np.ravel(Y).astype(np.float64)

        # bandwidth shape: (d,)
        self.bandwidth = self.get_bandwidth(self.X, self.Y, self.method, self.scale)

        # Cache for prediction
        self._compute_cache()

        return self

    def get_bandwidth(self, X, Y, method, scale=1.0):
        match method:
            case 'lscv': return bw_lscv(X, Y)
            case 'gcv': return bw_gcv(X, Y)
            case 'aicc': return bw_aicc(X, Y)
            case _: return super().get_bandwidth(X, method, scale)

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
        X = self.whiten_transform(X)
        X_scaled = X * self.inv_bw

        if not return_grad:
            sum_w, sum_wy, min_d2 = _get_kr(
                X_scaled, self.X_scaled, self.Y,
            )

            # mean shape: (m,)
            mean = sum_wy / sum_w

            if not return_dens:
                return mean

            # dens shape: (m,)
            shift_factor = _gaussian(min_d2)
            dens = sum_w * shift_factor

            if normalize:
                dens *= self.normalizer
            return mean, dens

        sum_w, sum_wy, min_d2, sum_wx, sum_wyx = _get_kr_grad(
            X_scaled, self.X_scaled, self.Y,
        )

        # mean shape: (m,)
        mean = sum_wy / sum_w

        # mean_grad shape: (m, d)
        mean_grad = sum_wyx
        mean_grad -= mean[:, None] * sum_wx
        mean_grad *= self.inv_bw / sum_w[:, None]

        if self.diag_cov:
            mean_grad /= self.L
        else:
            solve_triangular(self.L, mean_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

        output = [mean, mean_grad]

        if return_dens:
            # dens shape: (m,)
            shift_factor = _gaussian(min_d2)
            dens = sum_w * shift_factor

            # dens_grad shape: (m, d)
            dens_grad = sum_wx
            X_scaled *= sum_w[:, None]
            dens_grad -= X_scaled
            dens_grad *= self.inv_bw
            dens_grad *= shift_factor[:, None]

            if self.diag_cov:
                dens_grad /= self.L
            else:
                solve_triangular(self.L, dens_grad.T, trans='T', lower=True, check_finite=False, overwrite_b=True)

            if normalize:
                dens *= self.normalizer
                dens_grad *= self.normalizer

            output.append(dens)
            output.append(dens_grad)

        return output


# -----------------------------------------------------------
# Bandwidth Selection for KR
# -----------------------------------------------------------

def bw_lscv(X, Y):
    """
    Least-Squares Cross Validation
    """
    def loss(log_h):
        h = np.exp(log_h)
        mean, inv_dens = _get_kr_loo(X, Y, h)

        denominator = np.maximum(1.0 - inv_dens, 1e-8)

        residuals = (Y - mean) / denominator
        return np.mean(residuals ** 2)

    return optimize_bw(loss, bw_silverman(X))

def bw_gcv(X, Y):
    """
    Generalized Cross Validation
    GCV: MSE / (1 - tr(W)/n)^2
    """
    def loss(log_h):
        h = np.exp(log_h)
        mean, inv_dens = _get_kr_loo(X, Y, h)
        avg_inv_dens = inv_dens.mean()

        denominator = np.maximum(1.0 - avg_inv_dens, 1e-6)

        mse = np.mean((Y - mean) ** 2)
        return mse / (denominator ** 2)

    return optimize_bw(loss, bw_silverman(X))

def bw_aicc(X, Y):
    """
    Corrected AIC
    Better for small samples than standard AIC.
    AICc: log(MSE) + (1 + tr(W)/n) / (1 - (tr(W)+2)/n)
    """
    n = len(Y)

    def loss(log_h):
        h = np.exp(log_h)
        mean, inv_dens = _get_kr_loo(X, Y, h)
        sum_inv_dens = inv_dens.sum()

        numerator = 1.0 + sum_inv_dens / n
        denominator = np.maximum(1.0 - (sum_inv_dens + 2.0) / n, 1e-6)

        mse = np.maximum(np.mean((Y - mean) ** 2), 1e-20)
        return np.log(mse) + (numerator / denominator)

    return optimize_bw(loss, bw_silverman(X))
