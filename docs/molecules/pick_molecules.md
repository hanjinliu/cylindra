# Pick Molecules

In cryo-ET, particle picking is a critical step for subsequent subtomogram averaging and
structural analysis. Although there are many powerful automatic picking tools available
to this field, manual picking is still important for many cases.

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

## Advanced manual pickers

More advanced manual picking can be achieved using plugins, such as:

- [cylindra-guided-picker](https://github.com/hanjinliu/cylindra-guided-picker)
