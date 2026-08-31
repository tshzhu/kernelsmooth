import numpy as np
from .kernel_density import bw_scott, bw_silverman, _pdist_sq, optimize_bw
from .kernel_regression import KernelRegression


class LocalLinearRegression(KernelRegression):
    # Gaussian Local Linear Regression

    __slots__ = ()

    def __init__(self, bandwidth='lscv', diag_cov=False):
        """
        Args:
            bandwidth: Optional[ array_like | str | Tuple[str, array_like] ]
                - array_like (float/list/ndarray): Fixed bandwidth.
                - str: Method name ('scott', 'silverman', 'lscv', 'gcv', 'aicc', ...).
                - Tuple[str, float]: Method with multiplier (e.g., ('scott', 0.5)).

            diag_cov: bool
                - True: k(x, x') = exp(-0.5 * ||(x - x') / (sqrt(diag(Σ)) * h)||^2)
                - False: k(x, x') = exp(-0.5 * ((x - x') / h)^T Σ^-1 ((x - x') / h))
        """
        super().__init__(bandwidth=bandwidth, diag_cov=diag_cov)

    def get_bandwidth(self, X, Y, method, scale=1.0):
        # FIXME: fix successive calls with different methods
        match method:
            case None: # Fixed bandwidth
                bandwidth = self.bandwidth
            case 'scott': # Rule-of-thumb
                bandwidth = bw_scott(X, scale)
            case 'silverman':
                bandwidth = bw_silverman(X, scale)
            case 'lscv': # Optimization-based
                bandwidth = bw_lscv(X, Y)
            case 'gcv':
                bandwidth = bw_gcv(X, Y)
            case 'aicc':
                bandwidth = bw_aicc(X, Y)
            case _:
                raise NotImplementedError(f"Unsupported bandwidth method: {method}")
        return bandwidth

    def predict(self, X, return_dens=False, normalize=False,
                return_mean_grad=False, return_dens_grad=False):
        # FIXME: fix computation of gradient after whitening; fix order of params and outputs; improve efficiency
        """
        Solves (Z' Diag(k) Z) beta = Z' Diag(k) y
        """
        X = self._transform(X)
        m, dim = X.shape
        X_scaled = X * self.inv_bw

        # Weights Calculation (Gaussian Kernel)
        dists_sq = self._scaled_cdist_sq(X_scaled) # (M, N)
        K = np.exp(-0.5 * dists_sq)

        # diff: (X_j - x) -> Shape (M, N, dim)
        diff = (self.X[None, :, :] - X[:, None, :])
        K_diff = K[:, :, None] * diff

        # dens: Sum of weights (M,) -> Scalar part
        dens = np.einsum('ij->i', K)

        # K_X: Sum of w * (X_j - x) -> (M, dim) -> Vector part
        K_X = np.einsum('ijk->ik', K_diff)

        # K_XXt: Sum of w * (X_j - x)(X_j - x)^T -> (M, dim, dim) -> Matrix part
        K_XXt = np.einsum('ijk,ijl->ikl', K_diff, diff)

        # K_Y: Sum of w * y -> (M,)
        K_Y = K @ self.Y

        # K_XY: Sum of w * y * (X_j - x) -> (M, dim)
        K_XY = np.einsum('ijk,j->ik', K_diff, self.Y)

        # Design Matrix (M, dim+1, dim+1)
        # [ dens   K_X.T ]
        # [ K_X   K_XXt   ]
        design = np.empty((m, dim + 1, dim + 1))
        design[:, 0, 0] = dens
        design[:, 0, 1:] = design[:, 1:, 0] = K_X
        design[:, 1:, 1:] = K_XXt

        # Regularization for numerical stability
        design[:, range(dim + 1), range(dim + 1)] += 1e-10

        # Target Vector (M, dim+1, 1)
        # [ K_Y K_XY ]^T
        target = np.empty((m, dim + 1, 1))
        target[:, 0, 0] = K_Y
        target[:, 1:, 0] = K_XY

        # Solve Linear System
        betas = np.linalg.solve(design, target).squeeze(axis=-1)

        mean = betas[:, 0] # beta_0 is the estimated value

        if not (return_dens or return_mean_grad or return_dens_grad):
            return mean

        output = [mean,]

        if return_dens:
            output.append(dens * self.normalizer if normalize else dens)

        if return_mean_grad:
            # LLR computes gradient (slope) explicitly as beta_1...beta_d
            mean_grad = betas[:, 1:]
            output.append(mean_grad)

        if return_dens_grad:
            # Density Gradient: sum( K * (xi - x) / h^2 )
            dens_grad = K_X * self.inv_bw_sq
            output.append(dens_grad * self.normalizer if normalize else dens_grad)

        return output


# -----------------------------------------------------------
# Bandwidth Selection for LLR
# -----------------------------------------------------------

def _get_hat_stats(X, Y, h):
    n, dim = X.shape

    dists_sq = _pdist_sq(X / h)
    K = np.exp(-0.5 * dists_sq)

    diff = (X[None, :, :] - X[:, None, :])
    K_diff = K[:, :, None] * diff

    design = np.empty((n, dim + 1, dim + 1))
    design[:, 0, 0] = np.einsum('ij->i', K)
    design[:, 0, 1:] = design[:, 1:, 0] = np.einsum('ijk->ik', K_diff)
    design[:, 1:, 1:] = np.einsum('ijk,ijl->ikl', K_diff, diff)

    design[:, range(dim + 1), range(dim + 1)] += 1e-10

    target = np.empty((n, dim + 1, 2))

    # Compute Hat Diagonals (H_ii) in Column 0
    target[:, 0, 0] = 1.0
    target[:, 1:, 0] = 0.0

    # Compute Fitted Values (Y_hat) in Column 1
    target[:, 0, 1] = K @ Y
    target[:, 1:, 1] = np.einsum('ijk,j->ik', K_diff, Y)

    H_diag, Y_hat = np.linalg.solve(design, target)[:, 0, :].T

    return Y_hat, H_diag

def bw_lscv(X, Y):
    """
    Leave-One-Out Cross Validation
    """
    def loss(log_h):
        Y_hat, H_diag = _get_hat_stats(X, Y, np.exp(log_h))

        denominator = np.maximum(1.0 - H_diag, 1e-8)

        loo_residuals = (Y - Y_hat) / denominator
        return np.mean(loo_residuals ** 2)

    return optimize_bw(loss, bw_silverman(X))

def bw_gcv(X, Y):
    """
    Generalized Cross Validation
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
    """
    n = len(Y)

    def loss(log_h):
        Y_hat, H_diag = _get_hat_stats(X, Y, np.exp(log_h))
        tr_H = H_diag.sum()

        numerator = 1.0 + tr_H / n
        denominator = np.maximum(1.0 - (tr_H + 2.0) / n, 1e-6)

        mse = np.mean((Y - Y_hat) ** 2)
        return np.log(mse) + (numerator / denominator)

    return optimize_bw(loss, bw_silverman(X))
