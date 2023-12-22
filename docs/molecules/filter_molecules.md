# Filter Molecules by Features

Filtering refers to the process of selecting a subset of molecules based on given
criteria (not image processing such as Gaussian filter). This operation is very useful
for rich data analysis.

## Run Filtering

:material-arrow-right-thin-circle-outline: API: [`filter_molecules`][cylindra.widgets.main.CylindraMainWidget.filter_molecules]

:material-arrow-right-thin-circle-outline: GUI: `Molecules > Filter molecules`

This method uses one of the features tagged to the molecules to determine which
molecules will be filtered.

![Filter molecules](../images/filter_molecules.png){ loading=lazy, width=320px }

??? info "List of parameters"
    1. "layer" is the molecules layer that will be filtered.
    2. "predicate" uses `polars` expression to describe filtering conditions.
    3. If the source spline of the original molecules should be inherited to the
       filtered molecules, check "Inherit source spline".
    4. Check "Preview" to see the result of the filtering in the viewer. Preview
       will be updated when you change the predicate.

The filtered molecules will be added to the viewer as a new layer with suffix `"-Filt"`.

## Examples

### 1. Collect molecules from the tips

```python
ui.filter_molecules("Mole-0", predicate="col('position-nm') < 100")
ui.mole_layers["Mole-0-Filt"]
```

### 2. Collect molecules close to a point

Coordinates of the molecules can be accessed by column name `x`, `y`, and `z`.

```python
# square of distance from (10, 20, 15)
dist2 = (pl.col("z") - 10) ** 2 + (pl.col("y") - 20) ** 2 + (pl.col("x") - 15) ** 2

# molecules whose distance from (10, 20, 15) is less than 50
is_close = dist2 < 50 ** 2

ui.filter_molecules("Mole-0", predicate=is_close)
```
