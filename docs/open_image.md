# Start Cylindra

### Launch the GUI

You can launch the GUI application using `cylindra` command.

```shell
cylindra
```

!!! note

    The `cylindra` command is enabled only when you are in the virtual environment.
    For conda users, you can activate with `conda activate my-env`

After the startup, the `napari` viewer will be shown with the `cylindra` widget added
on the right side as a dock widget.

??? info "Launch programatically"

    You can use the [`start`][cylindra.core.start] function to launch the GUI.
    The GUI object is returned.

    ```python
    from cylindra import start

    ui = start()
    ```

`napari` has a [integrate IPython console](https://napari.org/stable/tutorials/fundamentals/quick_start.html#open-an-image) for running Python code. You can open it
with ++ctrl+shift+c++. If you launched the `napari` viewer from `cylindra`, following
variables are pushed to the console global namespace:

- `ui`: The currently opened `cylindra` main GUI object.
- `viewer`: The `napari` viewer object.
- `np`: `numpy` module.
- `ip`: `impy` module.
- `pl`: `polars` module.
- `plt`: `matplotlib.pyplot` module.
- `Path`: `pathlib.Path` class.


### Open Images

:material-arrow-right-thin-circle-outline: API: [`open_image`][cylindra.widgets.main.CylindraMainWidget.open_image].

:material-arrow-right-thin-circle-outline: GUI: `File > Open image` or ++ctrl+k++ &rarr; ++ctrl+o++.

In GUI, an open-image dialog is shown.

![open image dialog](images/open_image_dialog.png){ loading=lazy, width=400px }

In this dialog, you can configure how to open the image. Note that the image opened in
the viewer is **NOT the original image**. `cylindra` uses a binned and optionally
filtered image for visualization.

1. Click "Select file" to select the image file to open. tiff and mrc files are
   supported.
2. Set the appropriate pixel scale. You can click "Scan header" to automatically detect
   the pixel scale.
3. Set the tilt range used for calculating missing wedges.
4. Bin sizes used during your analysis. For example, setting to `[2, 4]` will start
   calculation of 2&times; and 4&times; binned images, which will be loaded into the
   memory, and leave the original image in the memory-mapped state (ready to be read
   in the future). The 4&times; binned image will be shown in the viewer as a
   reference. In the later analysis, you can switch between the original, 2&times;
   binned and 4&times; binned images.
5. Set the filter to apply to the image. The filter is applied to the reference image
   shown in the viewer, not to the original image.
6. If you want to load the original image into the memory, check "Load the entire image
   into memory".
7. You can preview the selected image by clicking "Preview". A preview window will be
   shown, which only loads separate image slices to accelerate the image loading.
8. Click "Open" to calculate the binning/filtering and show the reference image.

![](images\viewer_00_open_image.png){ loading=lazy }

After opening an image, you'll find three layers added to the viewer:

- `"Drawing Layer"`: a [Points layer](https://napari.org/stable/howtos/layers/points.html) used to manually draw splines.
- `"Splines"`: a [Points layer](https://napari.org/stable/howtos/layers/points.html) showing the registered splines.
- `<name of the tomogram>`: an [Image layer](https://napari.org/stable/howtos/layers/image.html) showing the reference image.

The `"Drawing Layer"` will be selected, with the "add points" mode activated by default.

Next step: [Fit Splines](fit_splines.md)
