# Inspect CFT Results

To clarify if the analysis is done correctly or to determine the parameters, you may
sometimes want to inspect the results of the CFT. This can be done by the spectra
inspector widget.

:material-arrow-right-thin-circle-outline: GUI: `Analysis > Open spectra inspector` or ++ctrl+k++ &rarr; ++ctrl+v++

In this window, you can see the local- and global-CFT power spectra of the existing
splines, and the estimated peak positions (red markers) if available. You can change the
current spline and bin size for inspection in this GUI.

![Inspection result](images/inspect_local_cft.png){ loading=lazy, width=570px }

The upsampled local-CFT is useful to check the sub-pixel peak positions.

![Upsampled Result](images/inspect_local_cft_upsampled.png){ loading=lazy, width=570px }

If the spline is not well fitted to the image, the peaks of the local power spectrum are
blurred, especially in the x (angular) direction.

![Bad result](images/inspect_local_cft_bad.png){ loading=lazy, width=570px }

!!! note
    This widget is also used to measure the lattice parameters of unknown structures.
    See [here](case_studies/custom_config.md) for more information.
