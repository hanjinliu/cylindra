from setuptools import setup, find_packages

with open("mtprops/__init__.py", encoding="utf-8") as f:
    line = next(f)
    VERSION = line.strip().split()[-1][1:-1]
      
setup(name="MTProps",
      version=VERSION,
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="BSD 3-Clause",
      packages=find_packages(),
      install_requires=[
            "impy-array>=1.24.2",
            "magic-class>=0.5.9",
            "numba>=0.54",
            "pyqtgraph>=0.12",
            "mrcfile>=1.3.0",
      ],
      python_requires=">=3.7",
      )