# The "Expressions"

In `cylindra`, molecules are processed largely depending on the [**expression** system
of `polars`](https://pola-rs.github.io/polars/user-guide/concepts/expressions/). An
expression is represented by a `polars.Expr` object. A `polars.Expr` object describes
a computation that can be applied to a `polars.DataFrame` object, but at the moment it
is created, it doesn't do any calculation.

Let's start with a simple example using `polars.DataFrame`.

```python
import polars as pl  # import polars module

# create a DataFrame object
df = pl.DataFrame(
    {
        "nth": [0, 1, 2, 3],
        "score": [0.8, 0.9, 0.4, 0.8]
    },
)
df
```

``` title="Output:"
shape: (4, 2)
┌─────┬───────┐
│ nth ┆ score │
│ --- ┆ ---   │
│ i64 ┆ f64   │
╞═════╪═══════╡
│ 0   ┆ 0.8   │
│ 1   ┆ 0.9   │
│ 2   ┆ 0.4   │
│ 3   ┆ 0.8   │
└─────┴───────┘
```

You can create an expression for a filtration predicate.

```python
pl.col("score")  # expression indicating the column named "score"
pl.col("score") > 0.7  # expression of "score is larger than 0.7"
df_filt = df.filter(pl.col("score") > 0.7)  # here the expression is evaluated
df_filt
```

``` title="Output:"
shape: (3, 2)
┌─────┬───────┐
│ nth ┆ score │
│ --- ┆ ---   │
│ i64 ┆ f64   │
╞═════╪═══════╡
│ 0   ┆ 0.8   │
│ 1   ┆ 0.9   │
│ 3   ┆ 0.8   │
└─────┴───────┘
```

## Expressions in `cylindra`

The expression system is very useful to describe how to process the molecules (or
sometimes splines). It is used in the following methods:

- [Set spline radius](../lattice_params.md#set-global-radius-manually)
- [Calculate molecule features](features.md#calculate-features)
- [Filter molecules](filter_molecules.md)
- [Average with filtered molecules](../alignment/conventional.md#subtomogram-averaging)
- [Average groups of molecules](../alignment/conventional.md#subtomogram-averaging)

In these methods, expressions are given as a `str`. `pl` and `col` is available in the
namespace.
