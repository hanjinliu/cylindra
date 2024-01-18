# Molecule Features

Molecules are tagged with features. For example, just after [mapping molecules from a spline](spline_to_molecules.md), features will be like this:

```python
layer = ui.mole_layers["Mole-0"]  # the molecules layer named "Mole-0"
layer.molecules.features
```

``` title="Output:"
shape: (286, 3)
┌─────┬───────┬─────────────┐
│ nth ┆ pf-id ┆ position-nm │
│ --- ┆ ---   ┆ ---         │
│ i64 ┆ i64   ┆ f64         │
╞═════╪═══════╪═════════════╡
│ 0   ┆ 0     ┆ 6.1548      │
│ 0   ┆ 1     ┆ 5.2079      │
│ 0   ┆ 2     ┆ 4.261       │
│ 0   ┆ 3     ┆ 3.3141      │
│ …   ┆ …     ┆ …           │
│ 21  ┆ 9     ┆ 83.7184     │
│ 21  ┆ 10    ┆ 82.7715     │
│ 21  ┆ 11    ┆ 81.8246     │
│ 21  ┆ 12    ┆ 80.8777     │
└─────┴───────┴─────────────┘
```

The molecule features are stored as a [`polars.DataFrame`](https://docs.pola.rs) object.
You can get any of the columns by `df["column-name"]`.

``` python
import matplotlib.pyplot as plt

df = layer.molecules.features
plt.scatter(df["pf-id"], df["score"])
plt.show()
```

## Paint Molecules by Features

:material-arrow-right-thin-circle-outline: GUI: `Molecules > View > Paint molecules` or ++ctrl+k++ &rarr; ++c++

This method updates the color of each molecule according to the value of a feature
column.

![paint_molecules](../images/paint_molecules.png){ width=320px, loading=lazy }

## Calculate Features

:material-arrow-right-thin-circle-outline: API: [`calculate_molecule_features`][cylindra.widgets.main.CylindraMainWidget.calculate_molecule_features]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Features > Calculate molecule features`

This method calculates features from the existing features. The input argument should
follow the [`polars expression`](expressions.md) syntax.

??? example "calculate molecule features"

    Following function call will add a new feature named "pf-is-odd" to the molecules
    layer, which indicates whether the molecule id is odd or not.

    ``` python
    ui.calculate_molecule_features(
        layer="Mole-0",
        column_name="pf-is-odd",
        expression="col('pf-id') % 2 == 1",
    )
    print(ui.mole_layers["Mole-0"].molecules.features["pf-is-odd"])  # print the feature
    ```

## Calculate Lattice Structures

:material-arrow-right-thin-circle-outline: API: [`calculate_lattice_structure`][cylindra.widgets.main.CylindraMainWidget.calculate_lattice_structure]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Features > Calculate lattice structure`

Some of the structural parameters equivalent to [CFT parameters](../lattice_params.md)
can be calculated from the molecule coordinates and spline parameters.

??? example "calculate spacing and twist"

    ``` python
    ui.calculate_lattice_structure(
        layer="Mole-0",
        props=["spacing", "twist"],
    )
    ```
