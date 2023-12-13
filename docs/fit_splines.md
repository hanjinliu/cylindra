# Fit Splines

All the analysis begin with the spline fitting.

??? info "Try it out with the demo data"
    Reconstructed, binned tomograms of short microtubules are available in the
    [GitHub repository](https://github.com/hanjinliu/cylindra/blob/main/tests/13pf_MT.tif), which have been used for testing the programs. You can download it to try
    following instructions.

## Draw Splines

API: [`register_path`][cylindra.widgets.main.CylindraMainWidget.register_path]

`cylindra` uses the built-in functionality of `napari` to place points first, and fit
the points with a spline by pushing ++f1++ or clicking the "register points" button in
the `cylindra` tool bar.

![draw_splines](images/draw_splines.gif){ loading=lazy, width=400px }

!!! note
    You can view the image from the different direction using the `napari` [viewer
    buttons](https://napari.org/stable/tutorials/fundamentals/viewer.html#viewer-buttons).

??? info "Auto-picking tool"
    The `cylindra` toolbar has an auto-picking tool. If you placed >1 points, you can
    extend it with a fast auto-centering function. You can also use keybinding ++f3++.

??? info "Configure splines"
    Each spline is tagged with a `SplineConfig` object, which describes the average
    feature of the cylindric structure it represents. Default values are optimized for
    microtubules. If you want to analyze other structures, see [here](configure.md).

## Fit Automatically

Manually drawn splines usually does not fit the cylindric structure well. `cylindra`
provides some automatic spline fitting using template-free image alignment based on
auto-correlation first introduced in [Blestel et al., 2009](https://ieeexplore.ieee.org/abstract/document/5193043), although the original method only applies to 2D images.

### Rough fitting

API: [`fit_splines`][cylindra.widgets.main.CylindraMainWidget.fit_splines]

`Splines > Fitting > Fit splines` is the method to roughly fit splines.

### Refine the fitting

API: [`refine_splines`][cylindra.widgets.main.CylindraMainWidget.refine_splines]

`Splines > Fitting > Refine splines` is the method to further refine the splines after
rough fitting. In this function, the lattice helical parameters at this moment are
determined by [CFT analysis](cft.md), and the average projection along the spline axis
is calculated. Each sub-volume along the spline is aligned to the average to update the
spline positions.

## Fit Manually

Sometimes spline fitting may fail, probably due to the poor quality of the image or the
fitting being affected by the nearby structures. In this case, you can carefully fit the
splines manually in a new window from `Splines > Fitting > Fit splines manually`, or shortcut ++ctrl+k++ &rarr; ++ctrl+slash++.

![fit_splines_manually](images/fit_splines_manually.png){ loading=lazy, width=400px }

In this window, you can left-click in the projection to select the center of the
structure. You can move along the spline by changing the "Position" box (++up++,
++down++), or go to other splines by changing the "Spline No." box (++left++,
++right++).

Next step: [Cylindric Fourier Transformation](cft.md)
