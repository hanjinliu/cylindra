# Cylindra

Cylindra is a Python module for image analysis of cylindric structures in cryo-ET.
Spectral analysis, simulation and subtomogram averaging of heterogenic cylindrical
structures are implemented.

### Index

#### Getting Started

- [Installation](installation.md)
- [Basics](basics.md)

#### In-depth Tutorials

- [Open an image](open_image.md)
- [Fit splines](fit_splines.md)
- [Measure lattice parameters](lattice_params.md)
- [Inspect CFT Results](inspect_results.md)
- [Load & Save Projects](project_io.md)
- [Custom Workflows](workflows.md)
- [Spline](spline/index.md)
- [Molecules](molecules/index.md)
- [Subtomogram Alignment and Averaging](alignment/index.md)
- [Working with Many Projects](batch/index.md)
- [Working with External Softwares](extern/index.md)
- [Tomogram Simulation](simulate.md)
- [Process Images](process_images.md)
- [Case Studies](case_studies/index.md)

#### Others

- [API Reference](api/index.md)

### Major Dependencies

These are the major dependencies of `cylindra`. You don't have to fully understand
them, but knowing them will help you to use `cylindra` more efficiently.

??? info "Show list"
    - [numpy](https://numpy.org/): Most of the basic mathmatical operations, such as
      trigonometric functions and vector operations are done with this library.
    - [scipy](https://www.scipy.org/): Some of the advanced functions, such as Fourier
      transformation, 3D rotation and spline fitting are done with this library.
    - [polars](https://pola.rs): A library for tabular data analysis, with beautifully
      designed API.
    - [napari](https://napari.org/): Multi-dimensional image viewer. Many of the
      visualization functions in `cylindra` are based on this library.
    - [magicgui](https://pyapp-kit.github.io/magicgui/): A powerful GUI generator for
      Python. This library "hides" the complexity of GUI programming, making the code
      very clean.
    - [magic-class](https://hanjinliu.github.io/magic-class/): An extension of
      `magicgui` that can convert Python classes into a multi-functional GUI.Multi-threading, macro recording, command palette and undoing are implemented in
      this library.
    - [impy](https://hanjinliu.github.io/impy/): A Python library for image processing.
    - [acryo](https://hanjinliu.github.io/acryo/): A Python library for parallel cryo-ET
      data analysis.

### Reference

If you find `cylindra` useful in your work, please consider citing our paper.

```
Heterogeneous local structures of the microtubule lattice revealed by cryo-ET and non-averaging analysis
Hanjin Liu, Hiroshi Yamaguchi, Masahide Kikkawa, Tomohiro Shima
bioRxiv 2024.04.30.591984; doi: https://doi.org/10.1101/2024.04.30.591984
```
