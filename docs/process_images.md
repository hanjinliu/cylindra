# Process Images

:material-arrow-right-thin-circle-outline: GUI: `File > Process images`

There are many tools available for processing tomographic images. `cylindra` also
provides basic functions for image processing.

![image_processor](images/image_processor.png){ width=320px, loading=lazy }

To simplify the path specification, the "suffix" parameter is available. When the
"input image" path changed, the "output image" path will be automatically filled with
the suffix. For example, if the input image is `path/to/image.mrc`, and the suffix is
"_processed", the output image will be `path/to/image_processed.mrc`.
