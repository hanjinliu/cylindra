# Transform Molecules

## Translation of Molecules

:material-arrow-right-thin-circle-outline: API: [`translate_molecules`][cylindra.widgets.main.CylindraMainWidget.translate_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Translate molecules`

Translation is a transformation of positional coordinates without changing the
orientation of the molecules. There are two types of translations.

1. **Internal translation**: The shifts $(dz, dy, dx)$ are applied to the internal
   coordinates. For example, internal translation $(1, 0, 0)$ of tubulin molecules
   results in a microtubule with larger radius.
2. **External translation**: The shifts $(dz, dy, dx)$ are applied to the external
   (world) coordinates. For example, external translation $(1, 0, 0)$ of tubulin
   molecules results in a microtubule that is shifted along the $z$-axis of the
   tomogram.

??? example "Kinesin binding sites"

    Following code prepares molecules for the kinesin binding sites, given a molecules
    layer named "Mole-0" is correctly initialized at the tubulin positions.

    ```python
    ui.sta.seam_search_manually(layer='Mole-0', location=0)
    ui.filter_molecules(layer='Mole-0', predicate='col("isotype-id") == 0')
    ui.translate_molecules(layers=['Mole-0-Filt'], translation=(2.0, 2.04, 0.0), internal=True)
    ```

## Rotation of Molecules

:material-arrow-right-thin-circle-outline: API: [`rotate_molecules`][cylindra.widgets.main.CylindraMainWidget.rotate_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Rotate molecules`

Rotation is a transformation of the orientation of the molecules without changing the
positional coordinates. Only internal rotation is implemented now.
