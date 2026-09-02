"""Cython-accelerated Gaussian local-linear regression."""

import numpy as np
from scipy.linalg import solve_triangular

from ._local_linear_regression import _get_llr, _get_llr_beta_grad, _get_llr_loo
from .kernel_regression import KernelRegression
from .tools import EPS, _gaussian, bw_silverman, optimize_bw


class LocalLinearRegression(KernelRegression):
    """Gaussian local-linear kernel regression.

    The local polynomial is fitted in whitened coordinates.  ``predict``
    returns the local intercept and, optionally, its analytic derivative, the
    kernel density and the density gradient in the original input coordinates.
    """

    __slots__ = ()

    def __init__(self, bandwidth="lscv", diag_cov=False):
        super().__init__(bandwidth=bandwidth, diag_cov=diag_cov)

    def fit(self, X, Y):
        """Fit the shared kernel state used by local-linear prediction."""

        X = np.atleast_2d(X)
        Y = np.ravel(Y)

        # Match the parent regression contract before any numerical work.
        if Y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and Y have incompatible sample counts: {X.shape[0]} and {Y.shape[0]}"
            )

        self.X = self.whiten_fit_transform(X)
        # The Cython kernels consume contiguous one-dimensional float64 targets.
        self.Y = Y.astype(np.float64)

        # Reuse all KDE/NWR bandwidth methods; only the regression-specific
        # cross-validation criteria below differ in how fitted values are formed.
        self.bandwidth = self.get_bandwidth(self.X, self.Y, self.method, self.scale)
        if np.ndim(self.bandwidth) == 0:
            self.bandwidth = np.full(self.X.shape[1], self.bandwidth, dtype=np.float64)
        else:
            self.bandwidth = np.asarray(self.bandwidth, dtype=np.float64).ravel()
            if self.bandwidth.size != self.X.shape[1]:
                raise ValueError(
                    f"Bandwidth dimension mismatch. Expected {self.X.shape[1]}, got {self.bandwidth.size}"
                )
        self._compute_cache()
        return self

    def get_bandwidth(self, X, Y, method, scale=1.0):
        """Resolve LLR cross-validation criteria or inherited kernel methods."""

        match method:
            case "lscv":
                return bw_lscv(X, Y)
            case "gcv":
                return bw_gcv(X, Y)
            case "aicc":
                return bw_aicc(X, Y)
            case _:
                return KernelRegression.get_bandwidth(self, X, Y, method, scale)

    def predict(self, X, return_dens=False, normalize=False, return_grad=False):
        """Predict the local intercept and optional analytic derivatives.

        For every query, Cython accumulates the weighted normal equations for
        an affine model in local offsets.  The public prediction is only the
        intercept ``beta_0``; internal slope coefficients are retained solely
        to evaluate the model and its analytic derivative.
        """

        X = self.whiten_transform(X)
        # design has shape (m, d+1, d+1); target has shape (m, d+1).
        design, target, min_d2 = _get_llr(X, self.X, self.Y, self.bandwidth)

        # Keep the first-order moments before solve() potentially overwrites
        # its input when a future NumPy implementation opts into overwrite.
        sum_w = design[:, 0, 0].copy()
        sum_wx = design[:, 0, 1:].copy()

        # A small ridge term makes each local normal equation solvable when the
        # effective neighborhood contains nearly collinear offsets.
        design[:, np.arange(design.shape[1]), np.arange(design.shape[1])] += EPS
        betas = np.linalg.solve(design, target[..., None]).squeeze(axis=-1)
        mean = betas[:, 0]

        if not return_grad and not return_dens:
            return mean

        if return_grad:
            # For A(x) beta(x) = b(x), implicit differentiation gives
            # d beta = A^-1 (d b - d A beta).  Solving A.T lambda = e_0 lets
            # Cython accumulate only d beta_0 without forming one derivative
            # matrix per input coordinate.
            rhs = np.zeros_like(betas)
            rhs[:, 0] = 1.0
            adjoint = np.linalg.solve(np.swapaxes(design, 1, 2), rhs[..., None]).squeeze(axis=-1)
            mean_grad = _get_llr_beta_grad(
                X,
                self.X,
                self.Y,
                self.bandwidth,
                betas,
                adjoint,
                min_d2,
            )
            # Convert d beta_0 / d x_white back to original coordinates.
            if self.diag_cov:
                mean_grad /= self.L
            else:
                solve_triangular(
                    self.L,
                    mean_grad.T,
                    trans="T",
                    lower=True,
                    check_finite=False,
                    overwrite_b=True,
                )

        output = [mean]
        if return_grad:
            output.append(mean_grad)

        if return_dens:
            # The design's leading element is the shifted kernel sum.  Restore
            # the common Gaussian factor used for numerical stability.
            shift_factor = _gaussian(min_d2)
            dens = sum_w * shift_factor

            dens_grad = None
            if return_grad:
                # sum_wx stores sum(K * (x_i - x)); multiplying by h^-2
                # yields the Gaussian density derivative before unwhitening.
                dens_grad = sum_wx * (self.inv_bw ** 2)
                dens_grad *= shift_factor[:, None]
                if self.diag_cov:
                    dens_grad /= self.L
                else:
                    solve_triangular(
                        self.L,
                        dens_grad.T,
                        trans="T",
                        lower=True,
                        check_finite=False,
                        overwrite_b=True,
                    )

            if normalize:
                dens *= self.normalizer
                if dens_grad is not None:
                    dens_grad *= self.normalizer

            output.append(dens)
            if dens_grad is not None:
                output.append(dens_grad)

        return tuple(output)


def _get_hat_stats(X, Y, bandwidth):
    """Return fitted values and hat diagonals for an LLR bandwidth."""

    bandwidth = np.asarray(bandwidth, dtype=np.float64).ravel()
    design, target = _get_llr_loo(X, Y, bandwidth)
    p = design.shape[1]
    design[:, np.arange(p), np.arange(p)] += EPS
    solved = np.linalg.solve(design, target)
    return solved[:, 0, 1], solved[:, 0, 0]


def bw_lscv(X, Y):
    """Select LLR bandwidths by leave-one-out squared error."""

    def loss(log_h):
        mean, hat_diag = _get_hat_stats(X, Y, np.exp(log_h))
        denominator = np.maximum(1.0 - hat_diag, 1e-8)
        return np.mean(((Y - mean) / denominator) ** 2)

    return optimize_bw(loss, bw_silverman(X))


def bw_gcv(X, Y):
    """Select LLR bandwidths by generalized cross-validation."""

    n = len(Y)

    def loss(log_h):
        mean, hat_diag = _get_hat_stats(X, Y, np.exp(log_h))
        denominator = np.maximum(1.0 - hat_diag.sum() / n, 1e-6)
        return np.mean((Y - mean) ** 2) / denominator**2

    return optimize_bw(loss, bw_silverman(X))


def bw_aicc(X, Y):
    """Select LLR bandwidths with the finite-sample corrected AIC."""

    n = len(Y)

    def loss(log_h):
        mean, hat_diag = _get_hat_stats(X, Y, np.exp(log_h))
        tr_h = hat_diag.sum()
        numerator = 1.0 + tr_h / n
        denominator = np.maximum(1.0 - (tr_h + 2.0) / n, 1e-6)
        mse = np.maximum(np.mean((Y - mean) ** 2), 1e-20)
        return np.log(mse) + numerator / denominator

    return optimize_bw(loss, bw_silverman(X))
