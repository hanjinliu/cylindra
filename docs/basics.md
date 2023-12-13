# Basics

Before start analyzing data, here's the basics of what you should know.

### Components in the Tomograms

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

### Coordinate System

Following the convention of `napari` and `acryo`, the 3D coordinates are stored in
(z, y, x) order.

??? "Why not (x, y, z)?"
    The (z, y, x) order is more natural in the mathematical perspective. A matrix is
    indexed by (row, column), which corresponds to (y, x). Therefore, if you have a
    3D array `arr` and a molecule at position `(z, y, x)`, you can access the value
    of the molecule by `arr[z, y, x]`.

### Manual and Programmatical Operations

Owing to `magicgui` and  `magic-class`, all the operations can be done either manually
or programmatically. For example, clicking the `File > Open image` item will open a
dialog for you to select an image file, scale and so on. This function can also be
called programmatically by `ui.open_image(...)`.

### Macro Recording

Since all the manual operations can be described as function calls, `magic-class`
automatically record all the operations you have done. Macro recording starts when
you opened an image and will be refreshed when another image is opened or the session
is initialized.
This "macro recording" feature will make your data analysis reproducible and shareable.

!!! note
    The term "macro" is a bit misleading here. "Macro" usually refers to a special
    language or function in many cases, but here it is just a Python script.

### Project Files

A session of analysis can be saved as a project file. A project file is a collection
of splines, molecules, their features, recorded macro script and so on. Once a project
is saved, you can resume the analysis or view the results by loading the project file.

A project can be saved as a directory, a zip file or a tar file. Format of the project
does not affect your later operations, as `cylindra` automatically handles the
differences.

### Working with Other Softwares

As data analyses of electron microscopy is a complicated process, you may need to
combine many softwares together. Currently, `cylindra` provides functions to read/write
`Spline` and `Molecules` objects from/to `IMOD` files. If you want other formats to be
supported, please open an issue or submit a pull request to the [repository](https://github.com/hanjinliu/cylindra).
