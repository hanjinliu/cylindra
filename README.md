# cylindra

`cylindra` is a GUI-integrated, GPU-supported cryo-ET image analysis tool for cylindric periodic structures such as microtubules.

## Installation

- From source

```
git clone git+https://github.com/hanjinliu/cylindra
```

To install using pip, run following commands under your desired environment.

```
cd cylindra
pip install -e .
```

To build in place, run following commands.

```
cd cylindra
python setup.py build_ext --inplace
```

## Implemented Functions

- Automatic/maual fitting of splines to cylindrical structures in 3D.
- Calculating structural parameters (radius, lattice spacing, skew angle, protofilament number and starting number like 13_3).
- Coordinate transformation along splines. Straightening and opening are included.
- Automatic/manual determination of polarity.
- Monomer mapping along splines.
- Subtomogram averaging and template-based alignment.
- Viterbi alignment.
- Microtubule seam search without any binding proteins.

## Usage

- From shell

```
cylindra
```

- From a Python interpreter

```python
from cylindra import start

ui = start()  # launch a napari viewer with a cylindra dock widget.
```

## Issues

If you encountered any bugs or have any requests, feel free to [report an issue](https://github.com/hanjinliu/cylindra/issues).
For better reproducibility, please copy your environments from `Others > cylindra info` and the recorded macro from `Others > Show full macro`.
