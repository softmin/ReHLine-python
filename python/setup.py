from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import eigency

extensions = [
    Extension(
        "L3solver",
        ["L3solver.pyx"],
        include_dirs=["."] + eigency.get_includes(),
    ),
]

setup(
    name="L3solver",
    version="0.0.0",
    ext_modules=cythonize(extensions),
    packages=["L3solver"],
)
