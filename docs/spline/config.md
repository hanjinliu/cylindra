# Spline Configurations

A spline is tagged with a [SplineConfig][cylindra.components.SplineConfig] object. This
object is practically used to annotate the spline with structural features. For example,
if a spline represent a microtubule, it is tagged with a `SplineConfig` object with
parameters appropriate for microtubules. These parameters are used in spline fitting,
CFT analysis, and so on.

Every time a new spline is created, it is initialized with the default `SplineConfig`.
Make sure the correct configuration is used before adding splines.

## List of Parameters

- Parameters used during CFT analysis
  - `npf_range` ... an (`int`, `int`) tuple indicating the minimum/maximum of the
    number of protofilaments.
  - `spacing_range` ... an (`float`, `float`) tuple indicating the minimum/maximum of
    longitudinal spacing between monomers (in nanometer).
  - `twist_range` ... an (`float`, `float`) tuple indicating the minimum/maximum of
    twist angle (in degree).
  - `rise_range` ... an (`float`, `float`) tuple indicating the minimum/maximum of
    rise angle (in degree).
  - `rise_sign` ... the sign of the rise angle. This parameter is redundant for the
    mathematically. It was introduced only to make the rise angle and the start number positive, like "13_3" instead of "13_-3".

- Other parameters
  - `clockwise` ... `"PlusToMinus"` or `"MinusToPlus"`. This parameter is used for
    automatic polarity determination. If the cross-sectional view of the cylindric
    structure in the plus-to-minus orientation has clockwise chirality, set this
    parameter to `"PlusToMinus"`.
  - `thickness_inner` ... distance between the inner surface and the peak of the radial
    profileradius (in nanometer). This parameter is used for construction of cylindric
    coordinate system.
  - `thickness_inner` ... distance between the outer surface and the peak of the radial
    profileradius (in nanometer). This parameter is used for construction of cylindric
    coordinate system.
  - `fit_depth` ... the depth of sub-volumes used for spline fitting region (in
    nanometer).
  - `fit_width` ... the width of sub-volumes used for spline fitting region (in
    nanometer).

## Change the Default Configuration

The configuration is optimized for microtubules by default. To analyze other structures,
you need to change the default configuration.

The default configuration is stored in the `default_config` property.

``` python
ui.default_config
```

``` title="Output"
SplineConfig(
	npf_range=Range(min=11, max=17),
	spacing_range=Range(min=3.9, max=4.3),
	twist_range=Range(min=-0.65, max=0.65),
	rise_range=Range(min=5.0, max=13.0),
	rise_sign=-1,
	clockwise='MinusToPlus',
	thickness_inner=2.8,
	thickness_outer=2.8,
	fit_depth=49.0,
	fit_width=40.0
)
```

### Manually set the parameters

:material-arrow-right-thin-circle-outline: GUI: `Splines > Config > Update default config`

![update_default_config](../images/update_default_config.png){ lazy loading, width=400px }

### Measuring the approximate parameters of unknown structures

:material-arrow-right-thin-circle-outline: GUI: `Analysis > Open spectra inspector`

![spectra_inspector](../images/spectra_inspector.png){ lazy loading, width=480px }

The widget for inspecting the power spectrum of the spline can also be used to measure
lattice parameters. After enabling "Select axial peak" mode, you can manally select the
peak position to measure the lattice parameters. After selecting the axial peak, you can
then "Select angular peak".

### Load presets

:material-arrow-right-thin-circle-outline: GUI: `Splines > Config > Load default config`

Presets are stored in the user directory. You can load the presets from the menu.

By default, presets for following biological components are available:

- F-actin
- Eukaryotic microtubule
- BtubAB (a bacterial microtubule)
- Tobacco mosaic virus

## Save the Configuration

:material-arrow-right-thin-circle-outline: GUI: `Splines > Config > Save default config`

You can save current default configuration as a preset with arbitrary name.
