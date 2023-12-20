# Viterbi Alignment

Viterbi Alignment is a constrained subtomogram alignment algorithm that was introduced
in our work. It imposes longitudinal constraints between neighboring molecules and
optimize the alignment score using the Viterbi algorithm.

Since Viterbi Alignment only consider the longitudinal constraints, all the molecules
are split into each protofilament and aligned independently in parallel.

## Run Viterbi Alignment on a Landscape

:material-arrow-right-thin-circle-outline: API: [`run_viterbi_on_landscape`][cylindra.widgets.sta.SubtomogramAveraging.run_viterbi_on_landscape]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Landscape > Run Viterbi alignment on landscape`

![Run Viterbi on a Landscape](../images/run_viterbi_on_landscape.png){ loading=lazy, width=400px }

??? info "List of Parameters"

    1. Select the landscape in the "landscape layer" combobox.
    2. "Longitudinal range (nm)" is the constrant of the longitudinal distance between
       neighboring molecules.
    3. "Maximum angle (deg)" is the another constraint. It is the maximum allowed
       angle between the spline tangent and the vector connecting the two molecules.

## Run Viterbi Alignment without Constructing a Landscape

:material-arrow-right-thin-circle-outline: API: [`align_all_viterbi`][cylindra.widgets.sta.SubtomogramAveraging.align_all_viterbi]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Viterbi Alignment`

![Run Viterbi](../images/align_all_viterbi.png){ loading=lazy, width=480px }
