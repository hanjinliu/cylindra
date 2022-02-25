from setuptools import setup, find_packages

with open("mtprops/__init__.py", encoding="utf-8") as f:
    line = next(f)
    VERSION = line.strip().split()[-1][1:-1]

with open("README.md", "r") as f:
    readme = f.read()
    
setup(
    name="MTProps",
    version=VERSION,
    description="Toolkit of local cylindrical Fourier transformation for tomography.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    license="BSD 3-Clause",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "impy-array>=1.25.3",
        "magic-class>=0.5.22",
        "numba>=0.54",
        "pyqtgraph>=0.12",
        "mrcfile>=1.3.0",
        "napari>=0.4.14",  # point shading
    ],
    python_requires=">=3.8",
    )