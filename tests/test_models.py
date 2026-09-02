import numpy as np
import pytest

from kernelsmooth import (
    KernelDensity,
    KernelRegression,
    LocalLinearRegression,
    NadarayaWatsonRegression,
)


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


def test_nadaraya_watson_alias():
    assert NadarayaWatsonRegression is KernelRegression


@pytest.mark.parametrize(
    "bandwidth, message",
    [
        (0.0, "strictly positive"),
        (-0.5, "strictly positive"),
        ([0.5, 0.0], "strictly positive"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        ([0.5, -np.inf], "finite"),
    ],
)
def test_fixed_bandwidth_must_be_finite_and_positive(bandwidth, message):
    with pytest.raises(ValueError, match=message):
        KernelDensity(bandwidth=bandwidth)

    model = KernelDensity(bandwidth=0.5)
    with pytest.raises(ValueError, match=message):
        model.set_bandwidth(bandwidth)
    assert model.bandwidth == 0.5


@pytest.mark.parametrize(
    "model, responses",
    [
        (KernelDensity(bandwidth=0.5), None),
        (KernelRegression(bandwidth=0.5), np.array([1.0])),
        (LocalLinearRegression(bandwidth=0.5), np.array([1.0])),
    ],
)
def test_fit_requires_at_least_two_samples(model, responses):
    X = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="at least 2 samples"):
        if responses is None:
            model.fit(X)
        else:
            model.fit(X, responses)


@pytest.mark.parametrize(
    "model_class",
    [KernelRegression, LocalLinearRegression],
)
def test_regression_fit_requires_matching_sample_counts(model_class):
    X = np.zeros((3, 2))
    model = model_class(bandwidth=0.5)

    with pytest.raises(ValueError, match="incompatible sample counts: 3 and 2"):
        model.fit(X, np.zeros(2))

    with pytest.raises(ValueError, match="incompatible sample counts: 3 and 4"):
        model.fit(X, np.zeros(4))


def test_boke_compatibility_contract():
    X, y, query = _data()
    model = KernelRegression(bandwidth=0.7).fit(X, y)

    mean = model.predict(query)
    mean_dens, density = model.predict(query, return_dens=True)
    mean_grad, gradient = model.predict(query, return_grad=True)
    mean_all, gradient_all, density_all, density_gradient_all = model.predict(
        query,
        return_dens=True,
        return_grad=True,
    )

    assert model.bandwidth.shape == (X.shape[1],)
    assert model.Y.shape == (X.shape[0],)
    assert model.pdf(query).shape == (len(query),)
    assert model.whiten_inverse_transform(np.zeros_like(query)).shape == query.shape
    assert mean.shape == (len(query),)
    assert mean_dens.shape == mean.shape == mean_grad.shape == mean_all.shape
    assert density.shape == density_all.shape == (len(query),)
    assert gradient.shape == gradient_all.shape == density_gradient_all.shape == query.shape


def test_local_linear_regression_shapes_and_finite_values():
    X, y, query = _data()
    for diag_cov in (False, True):
        model = LocalLinearRegression(bandwidth=0.7, diag_cov=diag_cov).fit(X, y)
        values = model.predict(query)
        values_grad = model.predict(query, return_grad=True)
        values_dens = model.predict(query, return_dens=True)
        values_all = model.predict(query, return_dens=True, return_grad=True)

        assert values.shape == (len(query),)
        assert values_grad[0].shape == values.shape
        assert values_grad[1].shape == query.shape
        assert values_dens[0].shape == values.shape
        assert values_dens[1].shape == values.shape
        assert values_all[0].shape == values.shape
        assert values_all[1].shape == query.shape
        assert values_all[2].shape == values.shape
        assert values_all[3].shape == query.shape
        for output in (*values_grad, *values_dens, *values_all):
            assert np.all(np.isfinite(output))

        eps = 1e-6
        finite_diff = np.empty_like(values_all[3])
        for dim in range(query.shape[1]):
            query_plus = query.copy()
            query_minus = query.copy()
            query_plus[:, dim] += eps
            query_minus[:, dim] -= eps
            finite_diff[:, dim] = (
                model.predict(query_plus, return_dens=True)[1]
                - model.predict(query_minus, return_dens=True)[1]
            ) / (2 * eps)
        np.testing.assert_allclose(values_all[3], finite_diff, rtol=2e-4, atol=2e-4)

        mean_finite_diff = np.empty_like(values_grad[1])
        for dim in range(query.shape[1]):
            query_plus = query.copy()
            query_minus = query.copy()
            query_plus[:, dim] += eps
            query_minus[:, dim] -= eps
            mean_finite_diff[:, dim] = (
                model.predict(query_plus) - model.predict(query_minus)
            ) / (2 * eps)
        np.testing.assert_allclose(values_all[1], mean_finite_diff, rtol=2e-4, atol=2e-4)


def test_local_linear_regression_recovers_affine_function():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 2))
    y = 1.5 - 2.0 * X[:, 0] + 0.75 * X[:, 1]
    query = rng.normal(size=(8, 2))

    model = LocalLinearRegression(bandwidth=0.8).fit(X, y)
    mean, mean_grad = model.predict(query, return_grad=True)

    np.testing.assert_allclose(mean, 1.5 - 2.0 * query[:, 0] + 0.75 * query[:, 1], atol=2e-2)
    expected_grad = np.broadcast_to(np.array([-2.0, 0.75]), mean_grad.shape)
    np.testing.assert_allclose(mean_grad, expected_grad, atol=0.15)
