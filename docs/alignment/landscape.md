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

![Construct a Correlation Landscape](../images/construct_landscape.png){ loading=lazy, width=400px }
