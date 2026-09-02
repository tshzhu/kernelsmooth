# kernelsmooth

`kernelsmooth` provides fast Gaussian kernel smoothers for NumPy arrays. The
core summations are implemented in Cython and the public estimators expose
both values and useful derivatives:

- Gaussian kernel density estimation (KDE)
- Nadaraya--Watson regression (NWR), also named `KernelRegression`
- Gaussian local-linear regression (LLR)
- Density gradients and analytic local-linear prediction gradients
- Diagonal or full-covariance whitening
- Fixed, rule-of-thumb, and cross-validated bandwidth selection

The package is independent of BOKE and can be installed and used on its own.

## Installation

Install the pre-built wheel selected for your platform and Python version:

```bash
python -m pip install kernelsmooth
```

The package currently targets Python 3.10 or newer. Wheels are built by the
GitHub Actions workflow for CPython on Linux, macOS, and Windows. If a wheel is
not available for a particular platform, installing from source requires a C
compiler, NumPy headers, and Cython (the build requirements declared in
`pyproject.toml`).

## Quick start

```python
import numpy as np

from kernelsmooth import (
    KernelDensity,
    KernelRegression,
    LocalLinearRegression,
    NadarayaWatsonRegression,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = np.sin(X[:, 0]) + 0.25 * X[:, 1]
query = np.zeros((8, 2))

# KDE: density values and gradients in the original coordinates.
kde = KernelDensity(bandwidth="silverman").fit(X)
density, density_grad = kde.pdf(query, normalize=True, return_grad=True)

# Nadaraya--Watson regression. The descriptive alias is the same class.
nwr = NadarayaWatsonRegression(bandwidth="silverman").fit(X, y)
mean, mean_grad, density, density_grad = nwr.predict(
    query,
    return_dens=True,
    normalize=True,
    return_grad=True,
)

# Local-linear regression. Its gradient output is the analytic derivative of
# the predicted local intercept.
llr = LocalLinearRegression(bandwidth="silverman").fit(X, y)
mean, mean_grad = llr.predict(query, return_grad=True)
```

All estimators accept a one-dimensional query array for a single point; it is
treated as shape `(1, d)`. For a batch of `m` points in `d` dimensions, query
arrays should have shape `(m, d)`.

Training data must contain at least two samples. Regression targets must have
exactly one value per training sample (`len(y) == len(X)`). Fixed scalar and
per-feature bandwidths must contain only finite, strictly positive values;
invalid inputs raise `ValueError` before whitening or Cython evaluation.

## Estimators

### Kernel density estimation

For training points `x_j`, KDE evaluates the Gaussian kernel sum

```text
K(x) = sum_j exp(-||x - x_j||² / 2)
```

The public method is:

```python
model = KernelDensity(bandwidth="mlcv", diag_cov=False).fit(X)
density = model.pdf(query)
density, density_grad = model.pdf(
    query,
    normalize=True,
    return_grad=True,
)
```

`pdf` returns an unnormalized kernel sum by default. `normalize=True` applies
the Gaussian, bandwidth, covariance, and `1/n` normalization factor. The
gradient has shape `(m, d)` and is always with respect to the original input
coordinates.

### Nadaraya--Watson regression

`KernelRegression` and `NadarayaWatsonRegression` are aliases for the same
Gaussian NWR estimator:

```text
m(x) = sum_j K(x, x_j) y_j / sum_j K(x, x_j)
```

```python
nwr = KernelRegression(bandwidth="lscv").fit(X, y)

mean = nwr.predict(query)
mean, density = nwr.predict(query, return_dens=True)
mean, mean_grad = nwr.predict(query, return_grad=True)
mean, mean_grad, density, density_grad = nwr.predict(
    query,
    return_dens=True,
    normalize=True,
    return_grad=True,
)
```

The return order is fixed. `normalize=True` affects only `density` and
`density_grad`; the regression mean and its gradient are unchanged.

### Local-linear regression

LLR fits a local affine model around each query point. For local offsets
`u_j = x_j - x`, it solves the weighted system

```text
argmin_(a, b) sum_j K(x, x_j) [y_j - a - bᵀ u_j]²
```

The intercept `a` is the local mean estimate and `b` is the local slope used
internally to fit it. The same optional density and density-gradient outputs
are available:

```python
llr = LocalLinearRegression(bandwidth="gcv").fit(X, y)

mean = llr.predict(query)
mean, mean_grad = llr.predict(query, return_grad=True)
mean, density = llr.predict(query, return_dens=True)
mean, mean_grad, density, density_grad = llr.predict(
    query,
    return_dens=True,
    normalize=True,
    return_grad=True,
)
```

For LLR, `return_grad=True` returns the analytic derivative of `beta_0(x)` with
respect to the original input coordinates. The internal local-polynomial slope
coefficients are not returned.

## Bandwidths

Every estimator accepts either a fixed bandwidth or a method name:

| Form | Meaning |
| --- | --- |
| `0.5` | One fixed bandwidth, expanded to every feature after fitting |
| `[0.5, 1.0]` | One fixed bandwidth per feature |
| `"scott"` | Scott rule-of-thumb bandwidth |
| `"silverman"` | Silverman/IQR bandwidth |
| `"median"` | Median pairwise-distance bandwidth |
| `"mlcv"` | KDE maximum-likelihood cross-validation |
| `"lscv"` | Regression least-squares cross-validation |
| `"gcv"` | Regression generalized cross-validation |
| `"aicc"` | Regression corrected AIC selection |

Rule-of-thumb methods can be multiplied by a scalar or per-feature scale:

```python
KernelDensity(bandwidth=("silverman", 0.5))
KernelRegression(bandwidth=("scott", [0.8, 1.2]))
```

Cross-validation optimizes in log-bandwidth space around a Silverman initial
value. Bandwidth selection is performed after whitening.

## Whitening and covariance modes

The `diag_cov` constructor flag controls the coordinate transform used before
kernel evaluation:

- `diag_cov=True`: divide each feature by its sample standard deviation.
- `diag_cov=False`: use a Cholesky factor of the full sample covariance,
  preserving correlations through a Mahalanobis geometry.

The fitted model exposes `bandwidth`, `L`, `X`, `X_scaled`, `inv_bw`, and
`normalizer` for inspection. `whiten_inverse_transform` maps whitened offsets
back to original-coordinate offsets, which is useful when sampling around a
fitted point.

## Performance model

KDE and NWR evaluate the kernel sums in Cython loops with the Python GIL
released. Their direct evaluation cost is `O(m*n*d)` for `m` queries, `n`
training samples, and `d` features. LLR accumulates a `(d + 1) × (d + 1)`
local design matrix for each query and therefore has an additional quadratic
dependence on `d`.

The implementation is CPU-based and exact for the selected training set. It
does not require CUDA or a system CUDA toolkit. For very large datasets,
batching query points is recommended to bound temporary memory use.

## Development

From a checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

The Cython extensions are declared in `setup.py`; `MANIFEST.in` includes their
`.pyx` sources so source distributions can be rebuilt. GitHub Actions uses
`cibuildwheel` to build and test platform-specific wheels.

## License

This project is released under the [MIT License](LICENSE).
