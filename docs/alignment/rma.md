# Restricted Mesh Annealing (RMA)

Restricted Mesh Annealing (RMA) is a constrained subtomogram alignment algorithm
that was introduced in our work. It imposes longitudinal and lateral constraints
between molecules and optimize the alignment score using simulated annealing.

## Run RMA on a Landscape

:material-arrow-right-thin-circle-outline: API: [`run_annealing_on_landscape`][cylindra.widgets.sta.SubtomogramAveraging.run_annealing_on_landscape]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Landscape > Run annealing on landscape`

![Run RMA on a Landscape](../images/run_annealing_on_landscape.png){ loading=lazy, width=400px }

??? info "List of Parameters"

    1. Select the landscape in the "landscape layer" combobox.
    2. "Longitudinal range (nm)" is the constrant of the longitudinal distance between
       neighboring molecules.
    3. "Lateral range (nm)" is the constrant of the lateral distance between
       neighboring molecules.
    4. "Maximum angle (deg)" is the another constraint. It is the maximum allowed
       angle between the spline tangent and the vector connecting the two molecules.
    5. "temperature time const" is the time constant of the simulated annealing. Larger
       value means slower annealing. `1.0` is usually a good value.
    6. "random seeds" is the seed values for the random number generator. The "Add"
       button will randomly add a new seed value.
    7. You can preview the distribution of the longitudinal/lateral distances by
       clicking the "Preview molecule network" button.

## Run RMA without Constructing a Landscape

:material-arrow-right-thin-circle-outline: API: [`align_all_annealing`][cylindra.widgets.sta.SubtomogramAveraging.align_all_annealing]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Simulated Annealing`

![Run RMA](../images/align_all_annealing.png){ loading=lazy, width=480px }
