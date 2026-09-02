"""Kernel density and local kernel regression models."""

from .kernel_density import KernelDensity
from .kernel_regression import KernelRegression
from .local_linear_regression import LocalLinearRegression

# Descriptive alias for the Gaussian Nadaraya--Watson estimator.
NadarayaWatsonRegression = KernelRegression

__all__ = [
    "KernelDensity",
    "KernelRegression",
    "NadarayaWatsonRegression",
    "LocalLinearRegression",
]
