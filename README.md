# kernelsmooth

`kernelsmooth` is the standalone kernel-smoothing package used by BOKE. It
provides Gaussian kernel density estimation (KDE) and Nadaraya--Watson kernel
regression (KR), with Cython-accelerated implementations and NumPy reference
implementations.

## Installation

Use a standard Python 3.12 virtual environment (Linux/macOS commands):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

For a normal PyPI installation, replace the final command with
`python -m pip install kernelsmooth`. This standalone package is intended to
be published to both PyPI and GitHub independently of BOKE.

## Usage

```python
import numpy as np
from kernelsmooth import KernelDensity, KernelRegression

X = np.random.default_rng(0).normal(size=(20, 2))
y = np.sum(X**2, axis=1)

kde = KernelDensity(bandwidth="silverman").fit(X)
density = kde.pdf(np.zeros((3, 2)))

kr = KernelRegression(bandwidth="silverman").fit(X, y)
prediction = kr.predict(np.zeros((3, 2)))
```

The `KernelDensity_np` and `KernelRegression_np` exports provide the NumPy
reference implementations for numerical checks.
