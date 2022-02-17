# MTProps

Quantify local/global properties of microtubule (or other cylindrically periodic structures) from tomograms.

## Implemented Functions

- Fitting splines to microtubules in 3D (automatically, or manually in GUI).
- 3D Refinement of splines.
- Measuring microtubule radius.
- Calculating structural parameters (pitch length, skew angle, protofilament number and starting number like 13_3).
- Coordinate transformation along splines. Straightening and opening are included.
- Determination of microtubule polarity by seeing projection images along splines.
- Monomer mapping and alignment.

## Usage

```python
from mtprops import start

ui = start() # launch a napari viewer with a MTProps dock widget.
```