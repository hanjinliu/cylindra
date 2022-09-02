# MTProps

`MTProps` is a GUI-integrated, GPU-supported cryo-ET image analysis tool for cylindric periodic structures such as microtubules.

## Installation

- From source

```
git clone git+https://github.com/hanjinliu/MTProps
```

## Implemented Functions

- Automatic/maual fitting of splines to microtubules in 3D.
- Calculating structural parameters (radius, lattice spacing, skew angle, protofilament number and starting number like 13_3).
- Coordinate transformation along splines. Straightening and opening are included.
- Automatic/manual determination of microtubule polarity.
- Monomer mapping along splines.
- Subtomogram averaging and template-based alignment.
- Microtubule seam search without any binding proteins.
- Viterbi alignment.

## Usage

- From shell

```
mtprops
```

- From a Python interpreter

```python
from mtprops import start

ui = start()  # launch a napari viewer with a MTProps dock widget.
```

## Issues

If you encountered any bugs or have any requests, feel free to [report an issue](https://github.com/hanjinliu/MTProps/issues).
For better reproducibility, please copy your environments from `Others > MTProps info` and the recorded macro from `Others > Show full macro`.
