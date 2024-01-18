# Split & Combine Molecules

## Split Molecules

:material-arrow-right-thin-circle-outline: API: [ui.split_molecules][cylindra.widgets.main.CylindraMainWidget.split_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Split molecules by features`

![split_molecules](../images/split_molecules.png){ width=240px, loading=lazy }

This method adds multiple `Molecules` to the viewer by splitting the existing `Molecules` by a feature column. The feature column should be a categorical column. The new `Molecules` will be named as `<old name>_<feature>`. For example, if the original `Molecules` is named `Mole-0` and the feature column is `pf-id`, which takes values `0`,
 `1` and so on, the new `Molecules` will be named `Mole-0_0`, `Mole-0_1` and so on.

## Combine Molecules

There are several ways to combine informations of multiple molecules.

### Concatenate multiple `Molecules` into a single `Molecules` object

:material-arrow-right-thin-circle-outline: API: [ui.concatenate_molecules][cylindra.widgets.main.CylindraMainWidget.concatenate_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Combine > Concatenate molecules`

This method concatenates positions, rotations and all the features of multiple `Molecules` objects into a single object. If each `Moelcules` object has different features, the missing features will be filled with null values.

![concatenate_molecules](../images/concatenate_molecules.png){ width=480px, loading=lazy }

### Merge `Molecules` positions, rotations and features

:material-arrow-right-thin-circle-outline: API: [ui.merge_molecule_info][cylindra.widgets.main.CylindraMainWidget.merge_molecule_info]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Combine > Merge molecule info`

This method create a new `Molecules` object by referencing the positions, rotations and features of other `Molecules` objects

![merge_molecule_info](../images/merge_molecule_info.png){ width=240px, loading=lazy }

### Copy `Molecules` feature into another

:material-arrow-right-thin-circle-outline: API: [ui.copy_molecules_features][cylindra.widgets.main.CylindraMainWidget.copy_molecules_features]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Combine > Copy molecules features`

This method simply copies the features of one `Molecules` object into another.

![copy_molecules_features](../images/copy_molecules_features.png){ width=480px, loading=lazy }
