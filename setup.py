import os
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Modified from hmmlearn (https://github.com/hmmlearn/hmmlearn/blob/main/setup.py).
class build_ext(build_ext):
    def finalize_options(self):
        from pybind11.setup_helpers import Pybind11Extension
        self.distribution.ext_modules[:] = [Pybind11Extension(
            "mtprops._cpp_ext", ["cpp/_viterbi.cpp"], cxx_std=11)]
        super().finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()


INFO = {}

with open("mtprops/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            INFO["version"] = line.strip().split()[-1][1:-1]
        if line.startswith("__author__"):
            INFO["author"] = line.strip().split()[-1][1:-1]
        if line.startswith("__email__"):
            INFO["author_email"] = line.strip().split()[-1][1:-1]


with open("README.md", "r") as f:
    readme = f.read()
    
setup(
    name="MTProps",
    description="Fourier analysis and subtomogram averaging of cylindrical structures",
    long_description=readme,
    long_description_content_type="text/markdown",
    **INFO,
    license="BSD 3-Clause",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"mtprops": ["**/*.pyi", "*.pyi"]},
    setup_requires=["pybind11>=2.9.2"],
    cmdclass={"build_ext": build_ext},
    py_modules=[],
    install_requires=[
        "impy-array>=2.0.0",
        "acryo>=0.0.2",
        "magicgui>=0.4.0",
        "magic-class>=0.6.4",
        "pyqtgraph>=0.12.4",
        "mrcfile>=1.3.0",
        "napari>=0.4.15",
    ],
    python_requires=">=3.8",
    ext_modules=[Extension("", [], language="c++")],
)