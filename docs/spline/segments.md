# Annotate Segments on Splines

Splines can be tagged with `SplineSegment` objects to represent segments along the
spline. A `SplineSegment` is defined by a start and end position along the spline, and
a value (e.g., a label or a measurement) associated with that segment.

Segments are designed to facilitate manual annotation along splines. For example, one
can annotate protein-bound regions, conformational states, etc.

## Manually Editing Segments

The most useful way to edit segments is through the [spline 3D interactor widget](3d_interaction.md). Added segments will be represented as yellow rings on the spline.

## Low-level API

:material-arrow-right-thin-circle-outline: API: [`add_segment`][cylindra.widgets.main.CylindraMainWidget.add_segment]
:material-arrow-right-thin-circle-outline: GUI: `Splines > Segments > Add segment`

This method adds a segment by specifying the start and end positions in nanometer along
the selected spline.

:material-arrow-right-thin-circle-outline: API: [`delete_segments`][cylindra.widgets.main.CylindraMainWidget.delete_segments]
:material-arrow-right-thin-circle-outline: GUI: `Splines > Segments > Delete segments`

This method deletes segments specified by the indices on the selected spline.
