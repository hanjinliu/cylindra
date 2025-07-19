# RELION

&rarr; [RELION Documentation](https://relion.readthedocs.io/en/release-5.0/)

## Utilities for Working with RELION

### Opening RELION Jobs

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.open_relion_job`][cylindra_builtins.relion.open_relion_job]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Open RELION job`

After [reconstruction of a tomogram in RELION](https://relion.readthedocs.io/en/release-5.0/STA_tutorial/ReconstructTomo.html), the next step is usually to pick particles from the tomogram.

In this method, you have to specify the path to the RELION `job.star` file, which
contains the description of the job.

Currently, following RELION jobs are supported:
- Reconstruct tomograms
- Denoise tomograms (predict)
- Pick tomograms
- Extract subotomos
- 3D initial reference
- 3D auto-refine

### Save Pick Results As A "Optimisation Set"

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.save_coordinates_for_import`][cylindra_builtins.relion.save_coordinates_for_import]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Save coordinates for import`

This method saves the current state of the `cylindra` batch analyzer, which contains
individual tomograms and multiple molecules, as an particle star file. This file can
directly be used for the [subtomogram extraction job in RELION](https://relion.readthedocs.io/en/release-5.0/STA_tutorial/ExtractSubtomos.html).

## Low-level File I/O
### Import Molecules from RELION

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.load_molecules`][cylindra_builtins.relion.load_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Load molecules`

This function read a `.star` file and add the contents to the viewer as molecules.

### Export Molecules for RELION

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.save_molecules`][cylindra_builtins.relion.save_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Save molecules`

This function saves the selected molecules as a `.star` file.

### Import Splines from RELION

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.load_splines`][cylindra_builtins.relion.load_splines]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Load splines`

The positional coordinates are used for drawing splines. The "rlnHelicalTubeID" column
is used to group the coordinates into each splines.

### Export Splines for RELION

:material-arrow-right-thin-circle-outline: API: [`cylindra_builtins.relion.save_splines`][cylindra_builtins.relion.save_splines]

:material-arrow-right-thin-circle-outline: GUI: `Plugins > RELION > Save splines`

`Spline` objects can be converted into segmented lines. This function saves the splines
as segmented lines in a `.star` file. Following the convention, "rlnHelicalTubeID" are
used as the column name for identifying each splines.

!!! warning
    Reading a `.star` file and saving back to a new `.star` file does **not** preserve the original data.
