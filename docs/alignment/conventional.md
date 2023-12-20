# Conventional Methods in Cryo-ET Studies

`cylindra` provides several common methods used in cryo-ET studies.

## Subtomogram Averaging

:material-arrow-right-thin-circle-outline: API: [`average_all`][cylindra.widgets.sta.SubtomogramAveraging.average_all]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Averaging > Average all molecules`

This method uses one or more molecules-layer to calculate the subtomogram average.

![Average all](../images/average_all.png){ loading=lazy, width=400px }

After running the method, a new `napari` viewer will be opened to show the average image. This viewer can be accessed via `ui.sta.sub_viewer`.

![Averaged tubulin](../images/average_tubulin.png){ loading=lazy, width=300px }

## Subtomogram Alignment

:material-arrow-right-thin-circle-outline: API: [`align_all`][cylindra.widgets.sta.SubtomogramAveraging.align_all]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Align all molecules`

![Align all](../images/align_all.png){ loading=lazy, width=400px }

## Align Averaged Image

:material-arrow-right-thin-circle-outline: API: [`align_averaged`][cylindra.widgets.sta.SubtomogramAveraging.align_averaged]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Align average to template`

This method is mostly used after molecules are initialized by [`map_monomers`](../molecules/spline_to_molecules.md#molecules-on-the-cylinder-surface).
It first calculates the average, then fit the average to the template image. The
optimal fitting parameters are then used to shift and rotate all the molecules. It
means that if you run `average_all` on the aligned molecules, the average image should
be almost well fitted to the template image.

![Align averaged](../images/align_averaged.png){ loading=lazy, width=400px }

## Fourier Shell Correlation (FSC)

:material-arrow-right-thin-circle-outline: API: [`calculate_fsc`][cylindra.widgets.sta.SubtomogramAveraging.calculate_fsc]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Analysis > Calculate FSC`

![Calculate FSC](../images/calculate_fsc.png){ loading=lazy, width=400px }

## PCA/K-means Classification

:material-arrow-right-thin-circle-outline: API: [`classify_pca`][cylindra.widgets.sta.SubtomogramAveraging.classify_pca]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Analysis > PCA/K-means classification`

![PCA/K-means classification](../images/classify_pca.png){ loading=lazy, width=400px }
