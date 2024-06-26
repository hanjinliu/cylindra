# Segmented Subtomogram Averaging (SSTA)

Segmented subtomogram averaging (SSTA) was first introduced by
[Guyomar et al](https://elifesciences.org/articles/83021). In the SSTA workflow,
subtomogram averaging is performed along the microtubules segment-by-segment so that
the resulting averages reflect the local structural heterogeneity.

In `cylindra`, SSTA can be easily performed owing to the powerful combination of the
[polars expression system](../molecules/expressions.md) and the
[built-in methods for cryo-ET](../alignment/conventional.md).

!!! note
    The EMPIAR deposition by the authors such as
    [EMPIAR-11253](https://www.ebi.ac.uk/empiar/EMPIAR-11253/) is very useful to test
    this workflow.

## Preparation

After loading the tomogram and the microtubules are properly fitted with splines,
molecules can be initialized along each spline at dimeric intervals. The interval can be
described by the `polars` expression `col('pitch') * 2`, which is evaluated with the
"pitch" value of spline global properties.

``` python
ui.infer_polarity(splines="all")
ui.global_cft_analysis(splines="all")  # to get lattice pitch
ui.map_along_spline(
    splines="all",
    molecule_interval="col('pitch') * 2",
    prefix="Center",
)
```

You can further refine the molecules by aligning them to the pre-determined structure
of microtubules such as [EMD-7974](https://www.ebi.ac.uk/emdb/EMD-7974).

!!! note
    The template image must be rotated in the correct direction so that the axis of
    the template microtubule matches the y-axis.

``` python
ui.sta.align_all(
    layers=['Center-0'],
    template_path="path/to/template.mrc",
    mask_params=(0.3, 0.8),
    max_shifts=(1.0, 1.0, 1.0),
    rotations=((0.0, 0.0), (10.0, 2.0), (0.0, 0.0)),
    cutoff=0.2,
)
```

By the standard subtomogram averaging using
[`average_all`][cylindra.widgets.sta.SubtomogramAveraging.average_all], you can check if
the alignment was successful.

``` python
ui.sta.average_all(layers=['Center-0-ALN1'], size=48.0)
```

## SSTA

Molecules can be grouped by its [features](../molecules/features.md). This means that
you can obtain averages for each group of molecules. The grouped averaging is a
generalization of what SSTA supposed to do. What we need here is to correctly label the
molecules for each segment, that is, a column with values like
`[0, 0, 0, 1, 1, 1, ..., n-1, n-1, n-1]` is needed supposing that there are `n`
segments.

Molecules generated by `map_along_spline` are already labeled with the `nth` column,
which takes the value of `[0, 1, ..., N-1]`. Expression `col('nth') // Ns` will group
the molecules into segments of `Ns` molecules. Since we initialized molecules at dimeric
intervals, the intervals are usually ~8.2 nm. For 100-nm SSTA, the `Ns` value should be
$100 / 8.2 \approx 12$.

``` python
ui.sta.average_groups(
    layers=['Center-0-ALN1'],
    size=48.0,
    by="col('nth') // 12",
)
```
