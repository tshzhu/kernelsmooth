import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


common_args = {
    "include_dirs": [numpy.get_include()],
    "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
}

extensions = [
    Extension(
        name="kernelsmooth._kernel_density",
        sources=["src/kernelsmooth/_kernel_density.pyx"],
        **common_args,
    ),
    Extension(
        name="kernelsmooth._kernel_regression",
        sources=["src/kernelsmooth/_kernel_regression.pyx"],
        **common_args,
    ),
    Extension(
        name="kernelsmooth._local_linear_regression",
        sources=["src/kernelsmooth/_local_linear_regression.pyx"],
        **common_args,
    ),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=False,
    ),
)
