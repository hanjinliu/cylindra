# IMOD

&rarr; [IMOD Home Page](https://bio3d.colorado.edu/imod/)

## Analyzing Tomograms Reconstructed by IMOD

### Open A Tomogram from An IMOD Project

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.open_image_from_imod_project`][cylindra_builtins.imod.open_image_from_imod_project]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Open image from an IMOD project`

This function uses the `.edf` files in an IMOD project to open a tomogram. Tilt angles
are automatically detected based on the files inside the project directory.
For the meaning of each parameter, see the [`open_image` function](../open_image.md#open-an-image).

### Open Multiple Tomograms from IMOD Projects

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.import_imod_projects`][cylindra_builtins.imod.import_imod_projects]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Import IMOD projects`

This function imports multiple tomograms from IMOD projects using wildcard paths or list
of paths, and adds them directly to the `cylindra` [batch analyzer](../batch/index.md).
You can click the "Open" button of the batch analyzer one by one to analyze each
tomogram.

## Create a `.prm` File for PEET

### Single Tomogram

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.export_project`][cylindra_builtins.imod.export_project]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Export project`

For subtomogram averaging and alignment, PEET uses a `.prm` file for the project
configuration. This function refers to the [STA widget](../alignment/index.md) and
creates a `.prm` file according to the parameters in the widget.

### Multiple Tomograms

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.export_project_batch`][cylindra_builtins.imod.export_project_batch]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Export project as batch`

## Low-level Functions

### Import Molecules from IMOD

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.load_molecules`][cylindra_builtins.imod.load_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Load molecules`

This function reads molecules from files in IMOD format, which are usually used in PEET.
Molecule positions are defined in a `.mod` file, while the molecule rotations are in a
CSV file. Additionally, offsets of each molecules may be recorded in the CSV file. By
passing the paths to these files, this function add molecules to the viewer.

### Export Molecules for IMOD

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.save_molecules`][cylindra_builtins.imod.save_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Save molecules`

This function saves the selected molecules as a `.mod` file and a CSV file in the same
directory. Saved files can be directly used in PEET.

### Import Lines as Splines from IMOD

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.load_splines`][cylindra_builtins.imod.load_splines]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Load splines`

In IMOD, you can create segmented lines with such as `3dmod` viewer and save as a
`.mod` file. The lines can be imported as splines in `cylindra`.

### Export Splines for IMOD

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.imod.save_splines`][cylindra_builtins.imod.save_splines]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > IMOD > Save splines`

`Spline` objects can be converted into segmented lines. This function saves the splines
as segmented lines in a `.mod` file.

!!! warning
    Reading a `.mod` file and saving back to a new `.mod` file does **not** preserve the
    original data.
