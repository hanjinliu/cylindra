# MTProps

`MTProps` is a GUI-integrated, GPU-supported cryo-ET image analysis tool for cylindric periodic structures such as microtubules.

## Installation

- From source

```
git clone git+https://github.com/hanjinliu/MTProps
```

## Implemented Functions

- Fitting splines to microtubules in 3D (automatically, or manually in GUI).
- Calculating structural parameters (radius, pitch length, skew angle, protofilament number and starting number like 13_3).
- Coordinate transformation along splines. Straightening and opening are included.
- Determination of microtubule polarity by seeing projection images along splines.
- Monomer mapping along splines.
- Subtomogram averaging and template-based alignment.

## Usage

- From a Python interpreter

```python
from mtprops import start

ui = start()  # launch a napari viewer with a MTProps dock widget.
```

- From console

```shell
python launch.py
```

## Issues

If you encountered any bugs or have any requests, feel free to [report an issue](https://github.com/hanjinliu/MTProps/issues).
For better reproducibility, please copy your environments from `Others > MTProps info` and the recorded macro from 
`Others > Create macro` if available.
