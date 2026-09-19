import os
import runpy
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.sdist import sdist
from setuptools_scm import get_version

__version__ = get_version(root=".", relative_to=__file__)

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# The directory that contains setup.py
SETUP_DIRECTORY = Path(__file__).resolve().parent

# Prepare before setuptools discovers license files and writes metadata. Git
# installs fetch missing headers; sdists verify their bundled copy offline.
if not os.environ.get("EIGEN3_INCLUDE_DIR"):
    runpy.run_path(str(SETUP_DIRECTORY / "tools" / "prepare_eigen.py"))["ensure_build_eigen"](SETUP_DIRECTORY)


class get_eigen_include:
    """Resolve local Eigen headers without accessing the network."""

    def __str__(self):
        override = os.environ.get("EIGEN3_INCLUDE_DIR")
        target = Path(override).expanduser().resolve() if override else SETUP_DIRECTORY / "vendor" / "eigen-5.0.1"
        if not (target / "Eigen" / "Core").is_file():
            raise RuntimeError(
                f"Eigen headers not found at {target}. Run 'python tools/prepare_eigen.py' before building from Git, "
                "or set EIGEN3_INCLUDE_DIR to a local directory containing Eigen/Core."
            )
        return str(target)


class checked_sdist(sdist):
    """Never emit a source distribution without the pinned Eigen dependency."""

    def run(self):
        prepare = runpy.run_path(str(SETUP_DIRECTORY / "tools" / "prepare_eigen.py"))
        target = prepare["ensure_build_eigen"](SETUP_DIRECTORY)
        # An external Eigen override may have skipped preparation while setup
        # metadata was read. The sdist still needs the pinned copy and licenses.
        licenses = [target / "LICENSE", *sorted(target.glob("COPYING*"))]
        self.distribution.metadata.license_files = [
            "LICENSE", *(path.relative_to(SETUP_DIRECTORY).as_posix() for path in licenses)
        ]
        super().run()


ext_modules = [
    Pybind11Extension(
        "rehline._internal",
        ["src/rehline.cpp"],
        include_dirs=[get_eigen_include(), "src"],
        depends=["src/rehline.h", "src/design.h"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="rehline",
    version=__version__,
    author=["Ben Dai", "Yixuan Qiu"],
    author_email="bendai@cuhk.edu.hk",
    url="https://rehline-python.readthedocs.io/en/latest/",
    description="Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence",
    packages=["rehline"],
    license_files=["LICENSE", "vendor/eigen-5.0.1/LICENSE", "vendor/eigen-5.0.1/COPYING*"],
    # install_requires=["requests", "pybind11", "numpy", "scipy", "scikit-learn"],
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext, "sdist": checked_sdist},
    zip_safe=False,
    python_requires=">= 3.10",
)

## build .so file
## $ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ./src/rehline.cpp -o _internal$(python3-config --extension-suffix)
