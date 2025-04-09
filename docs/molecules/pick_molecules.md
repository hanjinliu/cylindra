# Pick Molecules

In cryo-ET, particle picking is a critical step for subsequent subtomogram averaging and
structural analysis. Although there are many powerful automatic picking tools available
to this field, manual picking is still important for many cases.

In `cylindra`,

!!! note
    If you want to automatically generate molecules along splines rather than picking
    one by one, see [Spline to Molecules](spline_to_molecules.md).

## Manual pick from tomogram

:material-arrow-right-thin-circle-outline: API: [`register_molecules`][cylindra.widgets.main.CylindraMainWidget.register_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Register molecules`

As for [drawing splines](../fit_splines.md#draw-splines), manually added points in the
`napari` viewer can be converted into molecules with this method. Regardless of how the
points were added, this method will work properly without affecting macro recording, as
long as the points are added in the drawing layer. This means that you can use any
functionalities or `napari` plugins to pick molecules.

## Manual pick along splines

:material-arrow-right-thin-circle-outline: GUI: `Image > Open manual picker`

Because `cylindra` has many functions to manipulate splines, sometimes picking molecules
along splines will be more efficient. This is especially useful to look for binding
proteins along cylindrical structures, such as MAPs on microtubules.

The manual picker widget in `cylidnra` is designed for this purpose. It makes a "plane"
along the specified spline which can be moved, rotated or shifted relative to the
spline. You can imaging you set a camera on a drone and fly it in the spline orbit; you
can change the camera angles and focus to take pictures.

![Manual picker](../images/manual_picker.png){ loading=lazy, width=600px }

The center of the plane is first defined by the specified position on the spline. The
plane normal is parallel to the spline tangent vector, so that the plane is always
perpendicular to the spline by default. The plane is first rotated by roll, pitch and
yaw Euler angles, and focus offset is applied (shifted towards the direction defined by
the plane normal vector). Finally, tomogram is sliced by the user defined width and
depth.

You can click the 2D slice to add molecules. These molecules are linked to the `napari`
viewer; they are directly added to the `napari` drawing layer, and all the points in the
drawing layer will appear in the 2D slice if it is near the plane.
