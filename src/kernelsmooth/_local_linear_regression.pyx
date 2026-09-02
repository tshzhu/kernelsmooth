import numpy as np
cimport numpy as np
from libc.math cimport exp

ctypedef np.float64_t DTYPE_t


def _get_llr(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
    DTYPE_t[::1] Y_train,
    DTYPE_t[::1] bandwidth,
):
    """
    Accumulate local-linear weighted moments for prediction.

    The kernel is evaluated after dividing coordinate differences by
    ``bandwidth``, while the local polynomial is expressed in whitened
    (unscaled) coordinates.  A per-query
    minimum-distance shift is used for numerical stability; it multiplies the
    design matrix and target by the same constant and therefore leaves the
    fitted coefficients unchanged.

    Returns
    -------
    design : ndarray, shape (m, d + 1, d + 1)
        Weighted local-linear design matrices.
    target : ndarray, shape (m, d + 1)
        Weighted response moments.
    min_d2 : ndarray, shape (m,)
        Minimum squared bandwidth-scaled distance for each query.
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t p = d + 1
    cdef Py_ssize_t i, j, k, l

    cdef np.ndarray[DTYPE_t, ndim=3] design = np.zeros((m, p, p), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] target = np.zeros((m, p), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] min_d2 = np.zeros(m, dtype=np.float64)

    cdef DTYPE_t dist_sq, min_dist_sq, diff, weight, dx
    cdef DTYPE_t sw, swy
    cdef DTYPE_t *x = &X_test[0, 0]
    cdef DTYPE_t *train = &X_train[0, 0]
    cdef DTYPE_t *y = &Y_train[0]
    cdef DTYPE_t *bw = &bandwidth[0]
    cdef DTYPE_t *design_ptr = &design[0, 0, 0]
    cdef DTYPE_t *target_ptr = &target[0, 0]
    cdef DTYPE_t *min_ptr = &min_d2[0]

    with nogil:
        for i in range(m):
            min_dist_sq = 1.7976931348623157e+308

            # Find the nearest training point in bandwidth-scaled space.
            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = (x[k] - train[j * d + k]) / bw[k]
                    dist_sq += diff * diff
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

            min_ptr[i] = min_dist_sq
            sw = 0.0
            swy = 0.0

            # Accumulate moments using shifted kernel weights.
            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = (x[k] - train[j * d + k]) / bw[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * (dist_sq - min_dist_sq))
                sw += weight
                swy += weight * y[j]

                for k in range(d):
                    dx = train[j * d + k] - x[k]
                    design_ptr[i * p * p + k + 1] += weight * dx
                    design_ptr[i * p * p + (k + 1) * p] += weight * dx
                    target_ptr[i * p + k + 1] += weight * y[j] * dx

                    for l in range(d):
                        design_ptr[i * p * p + (k + 1) * p + l + 1] += (
                            weight * dx * (train[j * d + l] - x[l])
                        )

            design_ptr[i * p * p] = sw
            target_ptr[i * p] = swy

            x += d

    return design, target, min_d2


def _get_llr_loo(
    DTYPE_t[:, ::1] X,
    DTYPE_t[::1] Y,
    DTYPE_t[::1] bandwidth,
):
    """Accumulate unshifted local-linear moments for bandwidth selection."""

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t p = d + 1
    cdef Py_ssize_t i, j, k, l

    cdef np.ndarray[DTYPE_t, ndim=3] design = np.zeros((n, p, p), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] target = np.zeros((n, p, 2), dtype=np.float64)

    cdef DTYPE_t dist_sq, diff, weight, dx
    cdef DTYPE_t *x = &X[0, 0]
    cdef DTYPE_t *y = &Y[0]
    cdef DTYPE_t *bw = &bandwidth[0]
    cdef DTYPE_t *design_ptr = &design[0, 0, 0]
    cdef DTYPE_t *target_ptr = &target[0, 0, 0]

    with nogil:
        for i in range(n):
            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = (x[i * d + k] - x[j * d + k]) / bw[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * dist_sq)
                design_ptr[i * p * p] += weight
                target_ptr[i * p * 2 + 1] += weight * y[j]

                for k in range(d):
                    dx = x[j * d + k] - x[i * d + k]
                    design_ptr[i * p * p + k + 1] += weight * dx
                    design_ptr[i * p * p + (k + 1) * p] += weight * dx
                    target_ptr[i * p * 2 + (k + 1) * 2 + 1] += weight * y[j] * dx

                    for l in range(d):
                        design_ptr[i * p * p + (k + 1) * p + l + 1] += (
                            weight * dx * (x[j * d + l] - x[i * d + l])
                        )

            # The first target column yields the diagonal of the hat matrix.
            target_ptr[i * p * 2] = 1.0

    return design, target


def _get_llr_beta_grad(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
    DTYPE_t[::1] Y_train,
    DTYPE_t[::1] bandwidth,
    DTYPE_t[:, ::1] beta,
    DTYPE_t[:, ::1] adjoint,
    DTYPE_t[::1] min_d2,
):
    """Accumulate the analytic gradient of the local-linear intercept.

    Let ``z_j = [1, X_train[j] - x]`` and ``A = sum(w_j z_j z_j.T)``.
    With ``beta = A^{-1} b`` and ``lambda = A^{-T} e_0``, implicit
    differentiation gives, for coordinate ``k``::

        d beta_0 / d x_k = sum_j w_j * (
            u_jk / h_k**2 * r_j * (lambda.T @ z_j)
            - r_j * lambda[k + 1]
            + beta[k + 1] * (lambda.T @ z_j)
        )

    where ``u_j = X_train[j] - x`` and ``r_j = y_j - z_j.T @ beta``.
    The weights may be shifted by a query-specific constant because the
    corresponding design, target, and adjoint scale cancel algebraically.
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=2] grad = np.zeros((m, d), dtype=np.float64)

    cdef DTYPE_t *x = &X_test[0, 0]
    cdef DTYPE_t *train = &X_train[0, 0]
    cdef DTYPE_t *y = &Y_train[0]
    cdef DTYPE_t *bw = &bandwidth[0]
    cdef DTYPE_t *b = &beta[0, 0]
    cdef DTYPE_t *lam = &adjoint[0, 0]
    cdef DTYPE_t *md2 = &min_d2[0]
    cdef DTYPE_t *g = &grad[0, 0]

    cdef DTYPE_t dist_sq, diff, dx, weight
    cdef DTYPE_t fitted, residual, lambda_z, factor

    with nogil:
        for i in range(m):
            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = (x[k] - train[j * d + k]) / bw[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * (dist_sq - md2[i]))
                fitted = b[0]
                lambda_z = lam[0]
                for k in range(d):
                    dx = train[j * d + k] - x[k]
                    fitted += b[k + 1] * dx
                    lambda_z += lam[k + 1] * dx

                residual = y[j] - fitted
                for k in range(d):
                    dx = train[j * d + k] - x[k]
                    factor = (
                        dx / (bw[k] * bw[k]) * residual * lambda_z
                        - residual * lam[k + 1]
                        + b[k + 1] * lambda_z
                    )
                    g[k] += weight * factor

            x += d
            b += d + 1
            lam += d + 1
            g += d

    return grad
