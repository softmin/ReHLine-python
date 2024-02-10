import os
from pathlib import Path
import zipfile
import requests
from pybind11.setup_helpers import Pybind11Extension, build_ext
# from setuptools.command.build_ext import build_ext

__version__ = "0.0.3"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

SETUP_DIRECTORY = Path(__file__).resolve().parent

# Download Eigen source files
# Modified from https://github.com/tohtsky/irspack/blob/main/setup.py
class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:
        # Test whether the environment variable EIGEN3_INCLUDE_DIR is set
        # If yes, directly return this directory
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)
        if eigen_include_dir is not None:
            return eigen_include_dir

        # If the directory already exists (e.g. from previous setup),
        # directly return it
        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        # Filename for the downloaded Eigen source package
        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)
        # Unzip package
        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name

ext_modules = [
    Pybind11Extension("rehline._internal",
        sources=["src/rehline.cpp"],
        include_dirs=[get_eigen_include()],
        # Example: passing in the version to the compiled code
        # define_macros=[('VERSION_INFO', __version__)],
        ),
]

# class BuildFailed(Exception):
#     pass

# class ExtBuilder(build_ext):

#     def run(self):
#         try:
#             build_ext.run(self)
#         except (DistutilsPlatformError, FileNotFoundError):
#             raise BuildFailed('File not found. Could not compile C extension.')

#     def build_extension(self, ext):
#         try:
#             build_ext.build_extension(self, ext)
#         except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
#             raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmd_class": {"build_ext": build_ext}, "zip_safe": False}
    )