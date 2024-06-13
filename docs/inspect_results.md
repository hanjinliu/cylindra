# Inspect CFT Results

To clarify if the analysis is done correctly or to determine the parameters, you may
sometimes want to inspect the results of the CFT.

:material-arrow-right-thin-circle-outline: GUI: `Analysis > Open spectra inspector` or ++ctrl+k++ &rarr; ++ctrl+v++

In this window, you can see the local- and global-CFT power spectra of the existing
splines, and the estimated peak positions (red markers) if available.

![Inspect Results](images/inspect_local_cft.png){ loading=lazy, width=500px }

After clicking "Upsample spectrum" to enable the upsampling mode, you can upsample the
local power spectrum interactively by clicking the canvas.

![Inspect Results](images/inspect_local_cft_upsampled.png){ loading=lazy, width=500px }

!!! note
    This widget is also used to measure the lattice parameters of unknown structures.
    See [here](spline/config.md#measuring-the-approximate-parameters-of-unknown-structures)
    for more information.
