# Basics

Before start analyzing data, here's the basics of what you should know.

## Components

A `Tomogram` is a data structure with following information:

- Data source: this is not necessarily loaded into memory.
- Image metadata: voxel size, missing wedge, path to the data source, etc.
- Multiscale images: a list of images with different binning factors. For example,
  you can have a tomogram with 4&times;4&times;4 and 2&times;2&times;2 binned images
  and switch between them for different purposes (without actually saving them as
  separate image files).

A `Tomogram` is connected with following components:

- `Spline`: a piecewise cubic curve that represents the center line of a cylindric
  structure. A spline is defined by a set of the coefficients of the curves, so that
  it is very hard to directly interpret the internal data. However, you can easily
  sample points along the spline, differentiating the spline, and so on. In `cylindra`,
  many local properties and global properties are also stored in the `Spline` object.
- `Molecules`: a collection of molecules. This class is directly imported from
  [acryo](https://hanjinliu.github.io/acryo/main/molecules.html). A `Molecules` object
  is a collection of 3D points, 3D rotations and scalar features for each molecule.

## Coordinate Systems

Following the convention of `napari` and `acryo`, the 3D coordinates are stored in
(z, y, x) order.

??? "Why not (x, y, z)?"
    The (z, y, x) order is more natural in the mathematical perspective. A matrix is
    indexed by (row, column), which corresponds to (y, x). Therefore, if you have a
    3D array `arr` and a molecule at position `(z, y, x)`, you can access the value
    of the molecule by `arr[z, y, x]`.

Therefore, if you have a 3D image `img`, `img.shape[0]` is the size of z axis. If a list
of points is stored in a (N, 3) array `points`, `points[:, 0]` is the list of z
coordinates.

## Manual and Programmatic Operations

Owing to `magicgui` and `magic-class`, all the operations can be done either manually
or programmatically. For example, clicking the `File > Open image` item will open a
dialog for you to select an image file, scale and so on. This function can also be
called programmatically by `ui.open_image(...)`.

Throughout this documentation, you'll find following notations:

:material-arrow-right-thin-circle-outline: API: `ui.<method-name>`

:material-arrow-right-thin-circle-outline: GUI: `XXX > YYY` or maybe some shortcut keys

These notations tell you how to do the same thing programmatically or manually. You can
also open the command palette (++ctrl+p++) to search for the operation.

## Macro Recording

Since all the manual operations can be described as function calls, `magic-class`
automatically record all the operations you have done. Macro recording starts when
you opened an image and will be refreshed when another image is opened or the session
is initialized. This "macro recording" feature will make your data analysis
reproducible and shareable.

You can see the recorded macro script in `Others > Macro > Show macro` or by
++ctrl+shift+m++.

## Project Files

A session of analysis can be saved as a project file. A project file is a collection
of files needed to recover the current GUI state. Once a project is saved, you can
resume the analysis or view the results by loading the project file.

&rarr; [Learn more](project_io.md)

## Working with Other Softwares

As data analyses of electron microscopy is a complicated process, you may need to
combine many softwares together. `cylindra` provides functions to read or convert
components such as `Spline` and `Molecules` objects for other softwares. If you find
that some softwares would be nice to be supported but not currently, please open an
issue for the feature request or submit a pull request to the
[repository](https://github.com/hanjinliu/cylindra).

&rarr; [Learn more](extern/index.md)

## Configure Global Variables

:material-arrow-right-thin-circle-outline: GUI: `Others > Configure cylindra`

![Configure cylindra](images/configure_cylindra.png){ loading=lazy, width=400px }

Parameters in this dialog will be used across different sessions, but does not affect
the calculation results.
