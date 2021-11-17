# MTProps

Quantify local/global properties of microtubule (or other cylindrically periodic structures) from tomograms.

## Implemented Functions

- Fitting splines to microtubules in 3D.
- Measure microtubule radius.
- Calculate structural parameters (pitch length, skew angle, protofilament number and starting number like 13_3).
- Coordinate transformation along spline. Straightening and opening is included.
- Find seam position (WIP).
- Monomer mapping (WIP).
- Reconstruction (WIP).

## Usage

```python
from mtprops import start

ui = start()
```