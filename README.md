# kernelsmooth

Fast Gaussian kernel smoothers for NumPy arrays. The package provides
Cython-accelerated kernel sums and analytic derivatives for density estimation
and nonparametric regression.

- Gaussian kernel density estimation (`KernelDensity`)
- Nadaraya--Watson regression (`KernelRegression`, also
  `NadarayaWatsonRegression`)
- Gaussian local-linear regression (`LocalLinearRegression`)
- Full or diagonal covariance whitening
- Fixed, rule-of-thumb, and cross-validated bandwidth selection

## Installation

```bash
python -m pip install kernelsmooth
```

The package supports Python 3.10 and newer. Wheels are built for CPython on
Linux, macOS, and Windows. A source install requires a C compiler, NumPy
headers, and Cython.

## Quick start

```python
import numpy as np
from kernelsmooth import KernelDensity, NadarayaWatsonRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = np.sin(X[:, 0]) + 0.25 * X[:, 1]
X_new = np.zeros((8, 2))

kde = KernelDensity("silverman").fit(X)
density, density_grad = kde.pdf(X_new, normalize=True, return_grad=True)

nwr = NadarayaWatsonRegression("silverman").fit(X, y)
mean, mean_grad = nwr.predict(X_new, return_grad=True)
```

Training data use shape `(n, d)` and query points use shape `(m, d)`. A
one-dimensional query is treated as one point. Training requires at least two
samples; regression targets must satisfy `len(y) == len(X)`. Fixed bandwidths
must be finite and strictly positive.

## Estimators

### Kernel density estimation

For training points (x_j), the unnormalized Gaussian kernel sum is

$$
K(x) = \sum_{j=1}^{n} \exp\left(-\frac{1}{2}
\left\|\frac{x-x_j}{h}\right\|^2\right).
$$

```python
from kernelsmooth import KernelDensity

model = KernelDensity(bandwidth="mlcv", diag_cov=False).fit(X)
density = model.pdf(X_new)
density, density_grad = model.pdf(X_new, normalize=True, return_grad=True)
```

`pdf` returns the kernel sum by default. `normalize=True` applies the Gaussian,
bandwidth, covariance, and `1/n` normalization factor. The optional gradient
has shape `(m, d)` and is expressed in the original coordinates.

### Nadaraya--Watson regression

`KernelRegression` and `NadarayaWatsonRegression` are aliases. The estimator is

$$
\widehat m(x) =
\frac{\sum_{j=1}^{n} K(x,x_j)y_j}
{\sum_{j=1}^{n} K(x,x_j)}.
$$

```python
from kernelsmooth import KernelRegression

model = KernelRegression("lscv").fit(X, y)
mean = model.predict(X_new)
mean, density = model.predict(X_new, return_dens=True)
mean, mean_grad = model.predict(X_new, return_grad=True)
mean, mean_grad, density, density_grad = model.predict(
    X_new,
    return_dens=True,
    normalize=True,
    return_grad=True,
)
```

The full return order is `(mean, mean_grad, density, density_grad)`. Normalizing
the density does not change the regression mean or its gradient.

### Local-linear regression

For local offsets (u_j=x_j-x), LLR fits an affine model by weighted least
squares:

$$
(\widehat a,\widehat b) = \arg\min_{a,b}
\sum_{j=1}^{n} K(x,x_j)\left[y_j-a-b^\mathsf{T}u_j\right]^2.
$$

The prediction is the local intercept (widehat\beta_0(x)=\widehat a). The
optional `return_grad=True` output is the analytic derivative of this intercept
with respect to the input coordinates; the internal slope coefficients are not
returned.

```python
from kernelsmooth import LocalLinearRegression

model = LocalLinearRegression("gcv").fit(X, y)
mean = model.predict(X_new)
mean, mean_grad = model.predict(X_new, return_grad=True)
mean, density = model.predict(X_new, return_dens=True)
mean, mean_grad, density, density_grad = model.predict(
    X_new,
    return_dens=True,
    normalize=True,
    return_grad=True,
)
```

## Bandwidths

| Specification | Meaning |
| --- | --- |
| `0.5` | Fixed isotropic bandwidth |
| `[0.5, 1.0]` | Fixed per-feature bandwidths |
| `"scott"` | Scott rule of thumb |
| `"silverman"` | Robust Silverman/IQR rule |
| `"median"` | Median pairwise-distance bandwidth |
| `"mlcv"` | KDE maximum-likelihood cross-validation |
| `"lscv"` | Regression least-squares cross-validation |
| `"gcv"` | Regression generalized cross-validation |
| `"aicc"` | Regression corrected AIC |

Rule-of-thumb methods accept a scalar or per-feature multiplier:

```python
KernelDensity(("silverman", 0.5))
KernelRegression(("scott", [0.8, 1.2]))
```

Cross-validation searches in log-bandwidth space around a Silverman initial
value after whitening the predictors.

## Whitening

`diag_cov=True` standardizes each feature independently. The default
`diag_cov=False` uses a Cholesky factor of the full sample covariance and
preserves correlations in the Mahalanobis geometry.

Fitted models expose `bandwidth`, `L`, `X`, `X_scaled`, `inv_bw`, and
`normalizer`. `whiten_inverse_transform` maps whitened offsets back to original
coordinates.

## License

This project is released under the [MIT License](LICENSE).
