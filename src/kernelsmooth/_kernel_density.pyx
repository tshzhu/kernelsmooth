import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DTYPE_t


def _get_kde(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
):
    """
    dist^2 = || (xi/h) - (x/h) ||^2
    K = exp( -0.5 * dist^2 )
    dens = sum( K )

    Args:
        X_test: x/h, shape (m, d)
        X_train: xi/h, shape (n, d)

    Returns:
        sum_w: sum(K), shape (m,)
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(m, dtype=np.float64)

    cdef DTYPE_t *vec_x = &X_test[0, 0]
    cdef DTYPE_t *mat_xdata = &X_train[0, 0]

    cdef DTYPE_t *vec_sw = &sum_w[0]

    cdef DTYPE_t dist_sq, diff
    cdef DTYPE_t sw
    cdef DTYPE_t *vec_xi

    with nogil:
        for i in range(m):
            sw = 0.0
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                sw += exp(-0.5 * dist_sq)

                vec_xi += d

            vec_sw[i] = sw
            vec_x += d

    return sum_w


def _get_kde_grad(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
):
    """
    grad = sum( K * (xi - x) ) / h^2
         = ( sum( K * (xi/h) ) - (x/h) * sum( K ) ) / h

    Args:
        X_test: x/h, shape (m, d)
        X_train: xi/h, shape (n, d)

    Returns:
        sum_w: sum(K), shape (m,)
        sun_wx: sum(K * (xi/h)), shape (m,)
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] sum_wx = np.zeros((m, d), dtype=np.float64)

    cdef DTYPE_t *vec_x = &X_test[0, 0]
    cdef DTYPE_t *mat_xdata = &X_train[0, 0]

    cdef DTYPE_t *vec_sw = &sum_w[0]
    cdef DTYPE_t *vec_swx = &sum_wx[0, 0]

    cdef DTYPE_t dist_sq, diff, weight
    cdef DTYPE_t sw
    cdef DTYPE_t *vec_xi

    with nogil:
        for i in range(m):
            sw = 0.0
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * dist_sq)
                sw += weight

                for k in range(d):
                    vec_swx[k] += weight * vec_xi[k]

                vec_xi += d

            vec_sw[i] = sw

            vec_x += d
            vec_swx += d

    return sum_w, sum_wx


def _get_kde_loo(
    DTYPE_t[:, ::1] X,
    DTYPE_t[::1] bw,
):
    """
    dist^2 = || (xi/h) - (xj/h) ||^2
    K = exp( -0.5 * dist^2 )
    dens = sum( K )

    Args:
        X: xi, shape (n, d)
        bw: h, shape (d,)

    Returns:
        sum_w: sum(K), shape (n,)
    """

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(n, dtype=np.float64)

    cdef DTYPE_t *mat_x = &X[0, 0]
    cdef DTYPE_t *mat_xdata = <DTYPE_t *> malloc(n * d * sizeof(DTYPE_t))
    cdef DTYPE_t *vec_h = &bw[0]

    cdef DTYPE_t *vec_sw = &sum_w[0]

    cdef DTYPE_t dist_sq, diff, weight
    cdef DTYPE_t *vec_xi
    cdef DTYPE_t *vec_xj

    try:
        with nogil:
            # Scale data by bandwidth
            for i in range(n):
                for k in range(d):
                    mat_xdata[i * d + k] = mat_x[i * d + k] / vec_h[k]

            # Compute leave-one-out statistics
            for i in range(n):
                # Upper triangle (j > i)
                vec_xi = mat_xdata + i * d
                vec_xj = vec_xi + d

                for j in range(i + 1, n):
                    dist_sq = 0.0
                    for k in range(d):
                        diff = vec_xi[k] - vec_xj[k]
                        dist_sq += diff * diff

                    weight = exp(-0.5 * dist_sq)

                    vec_sw[i] += weight
                    vec_sw[j] += weight

                    vec_xj += d
    finally:
        free(mat_xdata)

    return sum_w
