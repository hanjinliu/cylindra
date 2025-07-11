# Restricted Mesh Annealing (RMA)

Restricted Mesh Annealing (RMA) is a constrained subtomogram alignment algorithm
that was introduced in our work. It imposes longitudinal and lateral constraints
between molecules and optimize the alignment score using simulated annealing.

As an example of using RMA, see the [case study](../case_studies/rma.md).

## Run RMA on a Landscape

:material-arrow-right-thin-circle-outline: API: [`run_rma_on_landscape`][cylindra.widgets.sta.SubtomogramAveraging.run_rma_on_landscape]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Landscape > Run annealing (RMA)`

![Run RMA on a Landscape](../images/run_rma_on_landscape.png){ loading=lazy, width=400px }

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

:material-arrow-right-thin-circle-outline: API: [`align_all_rma`][cylindra.widgets.sta.SubtomogramAveraging.align_all_rma]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Simulated Annealing`

![Run RMA](../images/align_all_rma.png){ loading=lazy, width=480px }

## Template-free RMA

:material-arrow-right-thin-circle-outline: API: [`align_all_rma_template_free`][cylindra.widgets.sta.SubtomogramAveraging.align_all_rma_template_free]

:material-arrow-right-thin-circle-outline: GUI: `Subtomogram Averaging > Alignment > Simulated annealing (RMA, template free)`

This method will iteratively construct landscape, align molecules by RMA, and validate
the result by FSC. This method is useful when you know that the structure of interest
is a cylindrical structure, but you do not know the monomer structure.
