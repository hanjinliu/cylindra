# RELION

&rarr; [RELION Documentation](https://relion.readthedocs.io/en/release-5.0/)

## Import Molecules from RELION

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.RELION.load_molecules`][cylindra.ext.relion.menu.RELION.load_molecules]

:material-arrow-right-thin-circle-outline: GUI: `File > RELION > Load molecules`

This function read a `.star` file and add the contents to the viewer as molecules.

## Export Molecules for RELION

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.RELION.save_molecules`][cylindra.ext.relion.menu.RELION.save_molecules]

:material-arrow-right-thin-circle-outline: GUI: `File > RELION > Save molecules`

This function saves the selected molecules as a `.star` file.

## Import Splines from RELION

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.RELION.load_splines`][cylindra.ext.relion.menu.RELION.load_splines]

:material-arrow-right-thin-circle-outline: GUI: `File > RELION > Load splines`

The positional coordinates are used for drawing splines. The "rlnHelicalTubeID" column
is used to group the coordinates into each splines.

## Export Splines for RELION

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.RELION.save_splines`][cylindra.ext.relion.menu.RELION.save_splines]

:material-arrow-right-thin-circle-outline: GUI: `File > RELION > Save splines`

`Spline` objects can be converted into segmented lines. This function saves the splines
as segmented lines in a `.star` file. Following the convention, "rlnHelicalTubeID" are
used as the column name for identifying each splines.

!!! warning
    Reading a `.star` file and saving back to a new `.star` file does **not** preserve the original data.
