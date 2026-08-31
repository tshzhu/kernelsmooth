import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport malloc, free

ctypedef np.float64_t DTYPE_t


def _get_kr(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
    DTYPE_t[::1] Y_train,
):
    """
    dist^2 = || (xi/h) - (x/h) ||^2
    K = exp( -0.5 * ( dist^2 - min( dist^2 ) ) )
    mean = sum( K * yi ) / sum( K )

    Args:
        X_test: x/h, shape (m, d)
        X_train: xi/h, shape (n, d)
        Y_train: yi, shape (n,)

    Returns:
        sum_w: sum(K), shape (m,)
        sum_wy: sum(K * yi), shape (m,)
        min_d2: min(dist^2), shape (m,)
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sum_wy = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] min_d2 = np.zeros(m, dtype=np.float64)

    cdef DTYPE_t *vec_x = &X_test[0, 0]
    cdef DTYPE_t *mat_xdata = &X_train[0, 0]
    cdef DTYPE_t *vec_ydata = &Y_train[0]

    cdef DTYPE_t *vec_sw = &sum_w[0]
    cdef DTYPE_t *vec_swy = &sum_wy[0]
    cdef DTYPE_t *ptr_md2 = &min_d2[0]

    cdef DTYPE_t dist_sq, min_dist_sq, diff, weight
    cdef DTYPE_t sw, swy
    cdef DTYPE_t *vec_xi

    with nogil:
        for i in range(m):
            # Find minimum squared distance
            min_dist_sq = 1.7976931348623157e+308 # DBL_MAX
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

                vec_xi += d

            ptr_md2[i] = min_dist_sq

            # Compute weighted sum
            sw = 0.0
            swy = 0.0
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * (dist_sq - min_dist_sq))

                sw += weight
                swy += weight * vec_ydata[j]

                vec_xi += d

            vec_sw[i] = sw
            vec_swy[i] = swy
            vec_x += d

    return sum_w, sum_wy, min_d2


def _get_kr_grad(
    DTYPE_t[:, ::1] X_test,
    DTYPE_t[:, ::1] X_train,
    DTYPE_t[::1] Y_train,
):
    """
    grad = ( sum( K * yi * (xi/h) ) - mean * sum( K * (xi/h) ) )
           / ( sum( K ) * h )

    Args:
        X_test: x/h, shape (m, d)
        X_train: xi/h, shape (n, d)
        Y_train: yi, shape (n,)

    Returns:
        sum_w: sum(K), shape (m,)
        sum_wy: sum(K * yi), shape (m,)
        min_d2: min(dist^2), shape (m,)
        sum_wx: sum(K * (xi/h)), shape (m,)
        sum_wyx: sum(K * yi * (xi/h)), shape (m,)
    """

    cdef Py_ssize_t m = X_test.shape[0]
    cdef Py_ssize_t n = X_train.shape[0]
    cdef Py_ssize_t d = X_test.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sum_wy = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] min_d2 = np.zeros(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] sum_wx = np.zeros((m, d), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] sum_wyx = np.zeros((m, d), dtype=np.float64)

    cdef DTYPE_t *vec_x = &X_test[0, 0]
    cdef DTYPE_t *mat_xdata = &X_train[0, 0]
    cdef DTYPE_t *vec_ydata = &Y_train[0]

    cdef DTYPE_t *vec_sw = &sum_w[0]
    cdef DTYPE_t *vec_swy = &sum_wy[0]
    cdef DTYPE_t *ptr_md2 = &min_d2[0]
    cdef DTYPE_t *ptr_swx = &sum_wx[0, 0]
    cdef DTYPE_t *ptr_swyx = &sum_wyx[0, 0]

    cdef DTYPE_t dist_sq, min_dist_sq, diff, weight
    cdef DTYPE_t sw, swy, yi
    cdef DTYPE_t *vec_xi

    with nogil:
        for i in range(m):
            # Find minimum squared distance
            min_dist_sq = 1.7976931348623157e+308 # DBL_MAX
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

                vec_xi += d

            ptr_md2[i] = min_dist_sq

            # Compute weighted sum
            sw = 0.0
            swy = 0.0
            vec_xi = mat_xdata

            for j in range(n):
                dist_sq = 0.0
                for k in range(d):
                    diff = vec_x[k] - vec_xi[k]
                    dist_sq += diff * diff

                weight = exp(-0.5 * (dist_sq - min_dist_sq))
                yi = vec_ydata[j]

                sw += weight
                swy += weight * yi

                for k in range(d):
                    ptr_swx[k] += weight * vec_xi[k]
                    ptr_swyx[k] += weight * yi * vec_xi[k]

                vec_xi += d

            vec_sw[i] = sw
            vec_swy[i] = swy

            vec_x += d
            ptr_swx += d
            ptr_swyx += d

    return sum_w, sum_wy, min_d2, sum_wx, sum_wyx


def _get_kr_loo(
    DTYPE_t[:, ::1] X,
    DTYPE_t[::1] Y,
    DTYPE_t[::1] bw,
):
    """
    dist^2 = || (xi/h) - (xj/h) ||^2
    K = exp( -0.5 * dist^2 )
    mean = sum( K * yi ) / sum( K )

    Args:
        X: xi, shape (n, d)
        Y: yi, shape (n,)
        bw: h, shape (d,)

    Returns:
        mean: sum(K * yi) / sum(K), shape (n,)
        inv_sum_w: sum(K), shape (n,)
    """

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] sum_w = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sum_wy = np.zeros(n, dtype=np.float64)

    cdef DTYPE_t *mat_x = &X[0, 0]
    cdef DTYPE_t *mat_xdata = <DTYPE_t *> malloc(n * d * sizeof(DTYPE_t))
    cdef DTYPE_t *vec_ydata = &Y[0]
    cdef DTYPE_t *vec_h = &bw[0]

    cdef DTYPE_t *vec_sw = &sum_w[0]
    cdef DTYPE_t *vec_swy = &sum_wy[0]

    cdef DTYPE_t dist_sq, diff, weight
    cdef DTYPE_t yi, yj
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
                vec_xi = mat_xdata + i * d
                yi = vec_ydata[i]

                # Diagonal (j = i)
                vec_sw[i] += 1.0
                vec_swy[i] += yi

                # Upper triangle (j > i)
                vec_xj = vec_xi + d

                for j in range(i + 1, n):
                    dist_sq = 0.0
                    for k in range(d):
                        diff = vec_xi[k] - vec_xj[k]
                        dist_sq += diff * diff

                    weight = exp(-0.5 * dist_sq)
                    yj = vec_ydata[j]

                    vec_sw[i] += weight
                    vec_swy[i] += weight * yj

                    vec_sw[j] += weight
                    vec_swy[j] += weight * yi

                    vec_xj += d
    finally:
        free(mat_xdata)

    cdef np.ndarray[DTYPE_t, ndim=1] mean = sum_wy / sum_w
    cdef np.ndarray[DTYPE_t, ndim=1] inv_dens = 1.0 / sum_w

    return mean, inv_dens
