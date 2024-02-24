# Clip and Invert Splines

The "clip" operation of a spline is **not** removing the points defining the spline.
Instead, it is equivalent to defining a linear transformation on the spline coordinate
$u$ as follows:

$\boldsymbol{C}_{clipped}(u) = \boldsymbol{C}(\alpha _{0} (1 - u) + \alpha _{1} u)$

$\boldsymbol{C}_{clipped}(0) = \boldsymbol{C}(\alpha _{0})$ and
$\boldsymbol{C}_{clipped}(1) = \boldsymbol{C}(\alpha _{1})$ indicate that the new spline
is defined by the $\alpha _{0} \le u \le \alpha _{1}$ region of the original spline.

As it is simply a linear transformation, spline inversion can also be defined the same
way:

$\boldsymbol{C}_{inv}(u) = \boldsymbol{C}(1 - u)$

## Clip by Length

:material-arrow-right-thin-circle-outline: API: [`clip_spline`][cylindra.widgets.main.CylindraMainWidget.clip_spline]

:material-arrow-right-thin-circle-outline: GUI: `Splines > Clip spline` or ++ctrl+k++ &rarr; ++ctrl+x++

This method clips the selected spline at both ends by the given lengths. Local
properties of the clipped spline will be discarded, while global properties will not.

![clip_spline](../images/clip_spline.png){ lazy loading, width=400px }

## Spline Clipper

:material-arrow-right-thin-circle-outline: GUI: `Splines > Open spline clipper`

![Spline Clipper](../images/spline_clipper.png){ lazy loading, width=480px }

For precise clipping, you can use the spline clipper widget. This widget synchronizes
the clipping lengths and the projections of the spline edges. "Clip here" will call
[`clip_spline`][cylindra.widgets.main.CylindraMainWidget.clip_spline] internally.


## Invert Splines

:material-arrow-right-thin-circle-outline: API: [`invert_spline`][cylindra.widgets.main.CylindraMainWidget.invert_spline]

:material-arrow-right-thin-circle-outline: GUI: `Splines > Orientation > Invert spline`

This method inverts the selected spline. This operation does not discard the local
properties. They will be inverted as well.
