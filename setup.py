import sys

sys.stderr.write(
    """
    ===============================================================
    cylindra does not support `python setup.py install`. Please use

        $ python -m pip install .

    instead.
    ===============================================================
    """
)
sys.exit(1)

from setuptools import setup, find_packages

CYLINDRA = "cylindra"

setup(
    name=CYLINDRA,
    license="BSD 3-Clause",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={CYLINDRA: ["**/*.pyi", "*.pyi", "**/*.png", "**/*.yaml"]},
    include_package_data=True,
    setup_requires=["pybind11>=2.10.4"],
    py_modules=[],
    install_requires=[
        "impy-array>=2.2.1",
        "acryo>=0.3.0",
        "magic-class>=0.6.16",
        "pyqtgraph>=0.12.4",
        "pyarrow>=11.0.0",
        "mrcfile>=1.3.0",
        "napari>=0.4.17",
        "qt-command-palette>=0.0.7",
    ],
    python_requires=">=3.9",
    extras_require={
        "tests": ["pytest", "pytest-qt"],
    },
    entry_points={
        "console_scripts": [f"{CYLINDRA}={CYLINDRA}.__main__:main"],
    },
)
