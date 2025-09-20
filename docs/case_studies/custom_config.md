# Prepare Spline Configuration for Your Structure of Interest

The default state of `cylindra` is optimized for microtubules. If you want to analyze
other cylindrical structures, you need to define a custom spline configuration. This is
a tedious task, but is very important for successful analysis.

## 1. Manually fit spline

To define a custom configuration, you will have to prepare a well-fitted spline first.

1. Draw a spline along your structure of interest. The length of the spline should be
   50 - 200 nm, depending on the density of your structure.
2. Open the spline fitter widget from `Splines > Open spline fitter`.
3. Click "Auto-center all" button. If the quality of your tomogram is good enough, you
   should see the center of the overlay accurately follows the center of your structure.
   If the auto-centering fails, you need to manually click the center of your structure
   slice by slice.
4. Click "Fit" button to apply the shift to the spline. This will update the spline
   coordinates.

## 2. Measure cylinder radius and thickness

Building a cylindrical coordinate system requires the radius and the thickness. Radius
can be measured for every spline, but the thickness values must be properly predefined
in the config.

1. Open the spline slicer widget from `Splines > Open spline slicer`. Because the radius
   is not known yet, no image will be shown in the canvas.
2. Click the "Measure radius" button. This will automatically measure the radius the
   spline by detecting the peak of the radial profile around the spline. Cross-sectional
   slice will be shown in the canvas, based on the radius. If the measured value largely
   deviates from your expectation, you can manually adjust the "Radius (nm)" value in
   the GUI.
3. Adjust the "Inner thickness" and "Outer thickness" parameters so that the circles
   shown in the canvas accurately represent the inner and outer surfaces of your
   structure. Cylindrical coordinate system will be constructed between these two
   surfaces.
4. Click "Apply radius and thickness" button to update spline radius and config.

## 3. Measure lattice parameters

Now, we can accurately build the cylindrical coordinate system to measure the lattice
parameters.

1. Open the spectra inspector widget from `Analysis > Open spectra inspector`.
2. By default, the global-CFT power spectrum of your structure is shown in the canvas.
   If you think the resolution of power spectrum is low, you can switch to "Upsampled
   global-CFT" mode.
3. Enable "Select axial peak" mode, and click the axial peak in the power spectrum.
4. Enable "Select angular peak" mode, and click the angular peak in the power spectrum.
5. The measured lattice parameters will be shown in the "Measured parameters" section.
   For later use, you can log these parameters to the console.

## 4. Determine config parameters

You now have lattice parameters of a representative segment of your structure. Based on
these values, you can determine the config parameters. Open the config editor widget
by following these steps:

1. Open the spline slicer widget.
   - `thickness_inner` and `thickness_outer` ... Set to the same value as the values you
     set in the spline slicer.
   - `clockwise` ... Click "Measure CW/CCW" button. This method will measure whether the
     current segment has a clockwise (CW) or counter-clockwise (CCW) slew and log the
     result in the logger widget. If the result is "clockwise" and the current segment
     visually appears to have the "PlusToMinus" polarity, set this parameter to
     "PlusToMinus". Do the same for other cases.

2. Open the spectra inspector widget.
   - `npf_range`, `spacing_range` and `twist_range` ... Set to a range that covers the
     measured parameters in the spectra inspector. There is no standard way to determine
     the range, as the actual achievable range depends on both the heterogeneity of the
     structure and the noise level of the reconstructed tomograms. The best way is to
     try several segments from different tomograms and get a sense of the distribution.
   - `rise_range` and `rise_sign` ... If the "rise" value is positive, leave `rise_sign`
     as is and set `rise_range` to a range that covers the measured value. If the "rise"
     value is negative, invert `rise_sign` and set `rise_range` to a range that covers
     the absolute value of the measured value. For example, if the measured "rise" is
     -10.5&deg; and "rise_sign" is -1, you can set `rise_sign` to 1 and `rise_range` to
     [8, 12].

!!! note "Range of nPF"

    Although in some cases the number of protofilaments (nPF) is well known to be a
    fixed value, it is recommended to set a small range. This will help you notice the
    bad fitting results.

!!! note "Sign of rise"

    The reason why we have a `rise_sign` parameter is that the `rise_angle` and `start`
    does not span the full minus to plus range for most type of structures, so we can
    just keep in mind that we always use the positive `rise_angle` and `start` &mdash;
    with `rise_sign` = 1, the lattice type of microtubule would be "13_-3", which is not
    what we usually see in the literature. Therefore, setting `rise_sign` does not have
    any mathematical meaning and will not affect the fitting result.

## 5. Save as a config preset

Click "Save as new config" button to save the current config as a new preset. You can
load this preset in the future analyses using the "Load preset" on the left panel.
