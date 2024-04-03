# Build Correlation Landscapes

Subtomogram alignment algorithms are usually implemented by maximizing the cross-
correlation between the subtomograms and a template image. Instead of searching for
the optimal parameter, we can calculate the correlation for all possible parameters
and visualize the correlation as a n-dimensional heatmap. This heatmap is reminiscent
of the "energy landscape" so we'll call it "correlation landscape".

## Construct a Correlation Landscape

:material-arrow-right-thin-circle-outline: API: [`construct_landscape`][cylindra.widgets.sta.SubtomogramAveraging.construct_landscape]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Landscape > Construct landscape`

This method will calculate the correlation landscape around selected molecules.

![Construct a Correlation Landscape](../images/construct_landscape.png){ loading=lazy, width=480px }

??? info "List of parameters"
    1. Set the template image and mask parameters in the STA widget.
    2. "layer" is the molecules layer that will be used to construct the landscape.
    3. "Max shifts (nm)" is the maximum allowed shift in (z, y, x) direction.
    4. "rotations" is the maximum allowed rotation angles in degree around each axis.
       Two `float` values are expected for each axis. `(a, b)` means the rotation
       angle will be between `-a` and `a` with step `b`. For example, `(4, 2)` means
       the rotation angles will be `-4`, `-2`, `0`, `2`, `4`.
    5. "cutoff" is the relative cutoff frequency of the low-pass filter.
    6. "interpolation" is the interpolation method used for resampling the sub-
       volumes.
    7. "upsample factor" defines how many times the correlation landscape will be
       upsampled. `5` means that the voxel size of the resulting landscape will be 1/5
       of the original image.

## Visualizing the Landscape

After the landscape construction, a `Landscape` layer, a subclass of [`Surface` layer](https://napari.org/stable/howtos/layers/surface.html)
will be added to the viewer. A `Landscape` layer shows the surface of an arbitrary
threshold. You can adjust the threshold value in the layer control.
