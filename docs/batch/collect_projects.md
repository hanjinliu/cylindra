# Collect projects

The collect_projects method collects projects into a `ProjectSequence` object, a
`list`-like container with `CylindraProject` objects.

```python
from cylindra import collect_projects

col = collect_projects("path/to/root/*.tar")
col
```

``` title="Output:"
ProjectSequence with 20 projects such as CylindraProject(...)
```

## Collect Spline Properties

To analyze all the splines properties in the projects, it is useful to collect them
into a single `polars.DataFrame` object. `ProjectSequence` has methods to do this.
These methods not only collect the properties, but add columns indicating which project
or spline the properties belong to.

- "spline-id": the index of the spline in the project
- "image-id": the index of the image in the project

### `collect_localprops`

This method collects all the local properties of the splines in the projects.
For example, following code plots the distribution of the "twist" parameters.

```python
import matplotlib.pyplot as plt

df = col.collect_localprops()
plt.hist(df["twist"])
```

Asides from the local properties, it also adds the following columns:

- "spl_pos": the spline coordinate (between 0 and 1) of the corresponding point in the
  spline. For example, if the local property is for the start point of the spline, this
  value is 0.0.
- "spl_dist": the distance (nm) of the corresponding point from the start of the spline.

### `collect_globalprops`

This method collects all the global properties of the splines in the projects.
For example, following code returns the number of splines with each orientation.

```python
df = col.collect_globalprops()
df["orientation"].value_counts()
```

``` title="Output:"
shape: (2, 2)
┌─────────────┬───────┐
│ orientation ┆ count │
│ ---         ┆ ---   │
│ str         ┆ u32   │
╞═════════════╪═══════╡
│ MinusToPlus ┆ 12    │
│ PlusToMinus ┆ 18    │
└─────────────┴───────┘
```

### `collect_joinedprops`

This method collects all the local and global properties. The column names of the global
properties will be suffixed with "_glob". The returned dataframe will have a lot of
duplicate values, but it is very useful for rich analysis. For example, following code
plots the distribution of the local "twist" parameters of 13-protofilament microtubules.

```python
df = col.collect_joinedprops()
plt.hist(df.filter(pl.col("npf_glob") == 13)["twist"])
```
