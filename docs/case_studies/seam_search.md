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

## Seam-search in the Tilt Series

:material-arrow-right-thin-circle-outline: API: [`seam_search_on_tilt`][cylindra.widgets.sta.SubtomogramAveraging.seam_search_on_tilt]

:material-arrow-right-thin-circle-outline: GUI: `STA widget > Analysis > Seam search > Seam search on tilt series`

If the tomogram is opened from a tilt series, the molecules can be scored directly in the
tilt images instead of the reconstructed subvolumes. The template (and the anti-template)
is projected to each tilt image at the position and orientation of each molecule, and
compared with the tilt image by ZNCC. The scores in each tilt are standardized over the
molecules (by the median and the median absolute deviation), and the score of each molecule
is their trimmed mean over the tilts (10% cut off from each end), which is robust to the noisy
high tilts and the outliers such as gold beads. The mean scores of the two groups of
molecules are compared for all the seam locations.
The probability of each seam location is estimated by bootstrapping the rings of molecules,
and the score of each protofilament (even rings minus odd rings) is plotted, whose sign
should flip at the seam.

??? info "List of parameters"

    1. "layer", the template, the mask, "anti-template path", "npf" and "cutoff" are the
       same as [`seam_search`](#seam-search-based-on-cross-correlation).
    2. "bin size" is the bin size of the projections.
    3. If "apply CTF" is checked, the CTF is applied to the projections, with the defocus
       of each molecule in each tilt. The tilt series must have the CTF parameters.
    4. "number of bootstrap samples" is the number of bootstrap samples to estimate the
       probability of each seam location. Set 0 to skip it.

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
