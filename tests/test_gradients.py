import numpy as np
from scipy.optimize import check_grad

from kernelsmooth import (
    KernelDensity,
    LocalLinearRegression,
    NadarayaWatsonRegression,
)


def _data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    Y = np.sin(X.sum(axis=1))
    points = rng.normal(scale=0.35, size=(5, X.shape[1]))
    return X, Y, points


def _check_density_gradient(model, point):
    def value(x, fitted):
        return float(fitted.pdf(x[None, :], normalize=True)[0])

    def gradient(x, fitted):
        return fitted.pdf(x[None, :], normalize=True, return_grad=True)[1][0]

    return check_grad(value, gradient, point, model, epsilon=1e-6)


def _check_regression_gradients(model, point):
    def mean_value(x, fitted):
        return float(fitted.predict(x[None, :])[0])

    def mean_gradient(x, fitted):
        return fitted.predict(x[None, :], return_grad=True)[1][0]

    def density_value(x, fitted):
        return float(
            fitted.predict(x[None, :], return_dens=True, normalize=True)[1][0]
        )

    def density_gradient(x, fitted):
        return fitted.predict(
            x[None, :],
            return_dens=True,
            normalize=True,
            return_grad=True,
        )[3][0]

    return (
        check_grad(mean_value, mean_gradient, point, model, epsilon=1e-6),
        check_grad(density_value, density_gradient, point, model, epsilon=1e-6),
    )


def test_all_public_gradient_paths_with_check_grad():
    X, Y, points = _data()

    for diag_cov in (False, True):
        kde = KernelDensity(bandwidth="silverman", diag_cov=diag_cov).fit(X)
        nwr = NadarayaWatsonRegression(
            bandwidth="silverman", diag_cov=diag_cov
        ).fit(X, Y)
        llr = LocalLinearRegression(
            bandwidth="silverman", diag_cov=diag_cov
        ).fit(X, Y)

        kde_errors = [_check_density_gradient(kde, point) for point in points]
        nwr_errors = [_check_regression_gradients(nwr, point) for point in points]
        llr_errors = [_check_regression_gradients(llr, point) for point in points]

        assert max(kde_errors) < 1e-5, kde_errors
        assert max(error[0] for error in nwr_errors) < 1e-5, nwr_errors
        assert max(error[1] for error in nwr_errors) < 1e-5, nwr_errors
        assert max(error[0] for error in llr_errors) < 1e-5, llr_errors
        assert max(error[1] for error in llr_errors) < 1e-5, llr_errors
