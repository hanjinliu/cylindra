from setuptools import setup, find_packages

with open("mtprops/__init__.py", encoding="utf-8") as f:
    line = next(f)
    VERSION = line.strip().split()[-1][1:-1]
      
setup(name="MTProps",
      version=VERSION,
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="GPLv2",
      packages=find_packages(),
      install_requires=[
            "scikit-image>=0.18",
            "numpy>=1.17",
            "scipy>=1.6.3",
            "matplotlib",
            "pandas>=1",
            "dask>=2021.6.0",
            "napari>=0.4.11",
      ],
      python_requires=">=3.7",
      )