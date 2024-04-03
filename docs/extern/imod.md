# IMOD

&rarr; [IMOD Home Page](https://bio3d.colorado.edu/imod/)

## Import Molecules from IMOD

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.IMOD.load_molecules`][cylindra.ext.imod.menu.IMOD.load_molecules]

:material-arrow-right-thin-circle-outline: GUI: `File > IMOD > Load molecules`

This function reads molecules from files in IMOD format, which are usually used in PEET.
Molecule positions are defined in a `.mod` file, while the molecule rotations are in a
CSV file. Additionally, offsets of each molecules may be recorded in the CSV file. By
passing the paths to these files, this function add molecules to the viewer.

## Export Molecules for IMOD

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.IMOD.save_molecules`][cylindra.ext.imod.menu.IMOD.save_molecules]

:material-arrow-right-thin-circle-outline: GUI: `File > IMOD > Save molecules`

This function saves the selected molecules as a `.mod` file and a CSV file in the same
directory. Saved files can be directly used in PEET.

## Import Lines as Splines from IMOD

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.IMOD.load_splines`][cylindra.ext.imod.menu.IMOD.load_splines]

:material-arrow-right-thin-circle-outline: GUI: `File > IMOD > Load splines`

In IMOD, you can create segmented lines with such as `3dmod` viewer and save as a
`.mod` file. The lines can be imported as splines in `cylindra`.

## Export Splines for IMOD

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.IMOD.save_splines`][cylindra.ext.imod.menu.IMOD.save_splines]

:material-arrow-right-thin-circle-outline: GUI: `File > IMOD > Save splines`

`Spline` objects can be converted into segmented lines. This function saves the splines
as segmented lines in a `.mod` file.

!!! warning
    Reading a `.mod` file and saving back to a new `.mod` file does **not** preserve the
    original data.

## Create a `.prm` File for PEET

:material-arrow-right-thin-circle-outline: API: [`ui.FileMenu.IMOD.export_project`][cylindra.ext.imod.menu.IMOD.export_project]

:material-arrow-right-thin-circle-outline: GUI: `File > IMOD > Export project`

For subtomogram averaging and alignment, PEET uses a `.prm` file for the project
configuration. This function refers to the [STA widget](../alignment/index.md) and
creates a `.prm` file according to the parameters in the widget.
