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

## Requirements

- **Python &ge; 3.9** (generic typings such as `list[int]` are used everywhere. If you have any requests on supporting lower version, please open an issue).
- (Optional) GPU and virtual environment in which the proper version of `cupy` is installed.
- **Sufficient memory size**. Most of the intense calculations are done out-of-core using `dask`, so that you can even run on 8-GB memory PC in many cases. However, larger memory size will make parallel processing more efficient. &ge;32 GB is recommended.

## Usage

- From shell

```shell
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
