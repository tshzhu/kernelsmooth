import numpy as np

from kernelsmooth import KernelDensity, KernelDensity_np, KernelRegression, KernelRegression_np


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3))
    y = np.sum(X**2, axis=1)
    query = rng.normal(size=(5, 3))
    return X, y, query


def test_kde_shapes_and_finite_values():
    X, _, query = _data()
    model = KernelDensity(bandwidth="silverman").fit(X)
    values, gradients = model.pdf(query, return_grad=True)
    assert values.shape == (len(query),)
    assert gradients.shape == query.shape
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))


def test_kr_shapes_and_finite_values():
    X, y, query = _data()
    model = KernelRegression(bandwidth="silverman").fit(X, y)
    values, gradients = model.predict(query, return_grad=True)
    assert values.shape == (len(query),)
    assert gradients.shape == query.shape
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))


def test_accelerated_and_numpy_models_agree():
    X, y, query = _data()
    kde = KernelDensity(bandwidth=0.7).fit(X)
    kde_np = KernelDensity_np(bandwidth=0.7).fit(X)
    kr = KernelRegression(bandwidth=0.7).fit(X, y)
    kr_np = KernelRegression_np(bandwidth=0.7).fit(X, y)
    np.testing.assert_allclose(kde.pdf(query), kde_np.pdf(query), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kr.predict(query), kr_np.predict(query), rtol=1e-10, atol=1e-10)
