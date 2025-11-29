# Spline

A 3D spline is a mapping from a 1D parameter to 3D points $(z, y, x)$:

$$
\boldsymbol{C}(u) \in \mathbb{R}^3 \quad (0\le u \le 1)
$$

Splines are used to represent filamentous structures in the tomograms. In `cylindra`,
several useful methods are provided to make the most of splines.

- [Clip and Invert Splines](clip.md)
- [Spline Configurations](config.md)
- [Annotate Segments on Splines](segments.md)

How to fit splines to structures is described [here](../fit_splines.md).
