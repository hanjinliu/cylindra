from pathlib import Path
from magicclass import (
    magicclass,
    MagicTemplate,
    set_options,
    vfield,
    confirm,
    set_design,
)
from magicclass.types import Optional, OneOf, SomeOf
from magicclass.ext.dask import dask_thread_worker
import impy as ip
from .widget_utils import FileFilter
from ._previews import view_image
from cylindra.const import GlobalVariables as GVar


@magicclass(record=False)
class ImageProcessor(MagicTemplate):
    """
    Process image files.

    Attributes
    ----------
    input_image : Path
        Path to the input image file.
    suffix : str, optional
        If given, `output_image` will be automatically updated to `input_image` value with suffix
        appended. For example, `input_image="path/to/image.mrc"` and `suffix="-0"` will fill
        `output_image` with `path/to/image-0.mrc`.
    output_image : Path
        Path where output image will be saved..
    """

    input_image = vfield(Path).with_options(filter=FileFilter.IMAGE)
    suffix = vfield(Optional[str]).with_options(value="-0", text="Do not autofill")
    output_image = vfield(Path).with_options(filter=FileFilter.IMAGE, mode="w")

    @input_image.connect
    @suffix.connect
    def _autofill_output_path(self):
        if self.suffix is None:
            return
        input_path = Path(self.input_image)
        output_path = input_path.with_stem(input_path.stem + self.suffix)

        n = 0
        while output_path.exists():
            output_path = output_path.with_stem(output_path.stem + f"-{n}")
        self.output_image = output_path

    def _confirm_path(self):
        return self.output_image.exists()

    @dask_thread_worker.with_progress(desc="Converting data type.")
    @set_design(text="Convert dtype")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def convert_dtype(self, dtype: OneOf["int8", "uint8", "uint16", "float32"]):
        """Convert data type of the input image."""
        img = self._imread(self.input_image)
        out = img.as_img_type(dtype)
        out.imsave(self.output_image)
        return None

    @dask_thread_worker.with_progress(desc="Inverting image.")
    @set_design(text="Invert intensity")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def invert(self):
        """Invert intensity of the input image."""
        img = self._imread(self.input_image)
        out = -img
        out.imsave(self.output_image)
        return None

    @dask_thread_worker.with_progress(desc="Low-pass filtering.")
    @set_design(text="Low-pass filter")
    @set_options(
        cutoff={"min": 0.05, "max": 0.85, "step": 0.05, "value": 0.5},
        order={"max": 20},
    )
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def lowpass_filter(self, cutoff: float, order: int = 2):
        """Apply Butterworth's tiled low-pass filter to the input image."""
        img = self._imread(self.input_image)
        out = img.tiled_lowpass_filter(cutoff, overlap=32, order=order)
        out.imsave(self.output_image)
        return None

    @dask_thread_worker.with_progress(desc="Binning.")
    @set_design(text="Binning")
    @set_options(
        bin_size={"min": 2, "max": 16, "step": 1},
    )
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def binning(self, bin_size: int = 4):
        """Bin image."""
        img = self._imread(self.input_image)
        out = img.binning(bin_size, check_edges=False)
        out.imsave(self.output_image)
        return None

    @dask_thread_worker.with_progress(desc="Flipping image.")
    @set_design(text="Flip image")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def flip(self, axes: SomeOf["x", "y", "z"] = ()):
        """Flip image by the given axes."""
        img = self._imread(self.input_image)
        for a in axes:
            img = img[ip.slicer(a)[::-1]]
        img.imsave(self.output_image)
        return None

    @set_design(text="Preview input image")
    def preview(self):
        """Open a preview of the input image."""
        view_image(self.input_image, self)
        return None

    def _imread(self, path, chunks=GVar.dask_chunk):
        return ip.lazy_imread(path, chunks=chunks)
