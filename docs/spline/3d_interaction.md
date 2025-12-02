# Spline 3D interactivity

*New in version 1.0.4*

Many spline operations require precise point selection along the spline in the context
of the tomogram and molecules. The spline 3D interactor widget injects mouse callback
functions to the drawing layer, allowing users to interactively select points on the
spline in 3D space.

:material-arrow-right-thin-circle-outline: GUI: `Splines > Open spline 3D interactor`

![](../images/spline_3d_interactor.png){ lazy loading, width=400px }

In this widget, you assign mouse left click and right click to "select spline" and "add
point on spline" actions, respectively.

![](../images/spline3d.gif){ lazy loading, width=480px }

- The sweep bar and local property plot in the spline control widget will update
  accordingly when you selected a point on the spline.
- After adding points on the spline, you can use the buttons below to add segments,
  delete segments, clip and split the spline.
- The value specified in "segment value" input box will be assigned to the new segments
  added by the "Add segments" button in this widget.
- The "Trim when splitting" input box specifies how much length to trim from both
  sides of the split point when splitting the spline.
