# Microtubule Seam-search

Microtubule seam-search is very important to separate &alpha;- and &beta;-tubulins.

What the seam-search algorithms in `cylindra` do is simply adding a new column named
"isotype-id" to the molecules features. As seam-search is usually combined with
subtomogram averaging and alignment, these methods are located in the STA widget.

## Seam-search Based on Cross-correlation

:material-arrow-right-thin-circle-outline: API: [`seam_search`][cylindra.widgets.sta.SubtomogramAveraging.seam_search]

:material-arrow-right-thin-circle-outline: GUI: `STA widget > Analysis > Seam search > Seam search by correlation`

Although the structure of &alpha;/&beta;-tubulins are very similar, it is known that
trying all the possible patterns and comparing the cross-correlation is useful to
distinguish them.

![Seam search](../images/seam_search.png){ loading=lazy, width=400px }

??? info "List of parameters"

    1. "layer" is the molecules layer to be seam-searched.
    2. Set the template image and mask parameters in the STA widget.
    3. "anti-template path" is the path to the anti-template image.
    4. "interpolation" is the interpolation method used for resampling the sub-
       volumes.
    5. "npf" is the number of protofilaments. If molecules are created in `cylindra`,
       this value should already be determined by [CFT](../lattice_params.md).
    6. You can select how to (or not to) show all the averages for different seam
       locations by the "show averages as" combobox.
    7. "cutoff" is the relative cutoff frequency of the low-pass filter. This value is
       usually smaller than the value used for subtomogram alignment.

Molecules labeled with `0` are the molecules that should be considered as the template,
and `1` the anti-template.

## Manual Seam-search

:material-arrow-right-thin-circle-outline: API: [`seam_search_manually`][cylindra.widgets.sta.SubtomogramAveraging.seam_search_manually]

:material-arrow-right-thin-circle-outline: GUI: `STA widget > Analysis > Seam search > Seam search manually`

This method labels the molecules by the given seam location. `location=0` will label
molecules with

```python
[0, 0, ..., 0,
 1, 1, ..., 1,
 0, 0, ..., 0,
 1, 1, ..., 1,]
```

and `location=1` will label molecules with

```python
[1, 0, ..., 0,
 0, 1, ..., 1,
 1, 0, ..., 0,
 0, 1, ..., 1,]
```

## Seam-search by Features

:material-arrow-right-thin-circle-outline: API: [`seam_search_by_feature`][cylindra.widgets.sta.SubtomogramAveraging.seam_search_by_feature]

:material-arrow-right-thin-circle-outline: GUI: `STA widget > Analysis > Seam search > Seam search by feature`

This method labels the molecules by the given feature. The feature should be such that
approximately label the &alpha;- or &beta;-tubulins. Usually, molecules should be
labeled by classification using microtubule-associated proteins as the fiducials.
