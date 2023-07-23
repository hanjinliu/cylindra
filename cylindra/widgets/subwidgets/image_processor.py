from typing import Annotated, Literal
import psutil
from pathlib import Path
from magicclass import (
    magicclass,
    MagicTemplate,
    set_options,
    vfield,
    confirm,
    set_design,
)
from magicclass.types import Optional
from magicclass.ext.dask import dask_thread_worker
import impy as ip
from cylindra.widgets.widget_utils import FileFilter
from cylindra.widgets._previews import view_image
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.const import GlobalVariables as GVar


@magicclass(record=False)
class ImageProcessor(MagicTemplate):
    """
    Process image files.

    Attributes
    ----------
    max_gb: float
        Maximum GB to load into memory. If the input image is larger than this value, the image
        will be processed lazily. Use the half of total memory by default.
    input_image : Path
        Path to the input image file.
    suffix : str, optional
        If given, `output_image` will be automatically updated to `input_image` value with suffix
        appended. For example, `input_image="path/to/image.mrc"` and `suffix="-0"` will fill
        `output_image` with `path/to/image-0.mrc`.
    output_image : Path
        Path where output image will be saved..
    """

    max_gb = vfield(psutil.virtual_memory().total * 5e-10, label="Max GB")
    input_image = vfield(Path).with_options(filter=FileFilter.IMAGE)
    suffix = vfield(Optional[str]).with_options(value="-output", text="Do not autofill")
    output_image = vfield(Path).with_options(filter=FileFilter.IMAGE, mode="w")

    _InputPath = Annotated[Path, {"widget_type": "EmptyWidget", "bind": input_image}]
    _OutputPath = Annotated[Path, {"widget_type": "EmptyWidget", "bind": output_image}]

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
    def convert_dtype(
        self,
        input: _InputPath,
        output: _OutputPath,
        dtype: Literal["int8", "uint8", "uint16", "float32"],
    ):
        """Convert data type of the input image."""
        img = self._imread(input)
        out = img.as_img_type(dtype)
        out.imsave(output)
        return None

    @dask_thread_worker.with_progress(desc="Inverting image.")
    @set_design(text="Invert intensity")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def invert(
        self,
        input: _InputPath,
        output: _OutputPath,
    ):
        """Invert intensity of the input image."""
        img = self._imread(input)
        out = -img
        out.imsave(output)
        return None

    @dask_thread_worker.with_progress(desc="Low-pass filtering.")
    @set_design(text="Low-pass filter")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def lowpass_filter(
        self,
        input: _InputPath,
        output: _OutputPath,
        cutoff: Annotated[float, {"min": 0.05, "max": 0.85, "step": 0.05}] = 0.5,
        order: Annotated[int, {"max": 20}] = 2,
    ):
        """Apply Butterworth's tiled low-pass filter to the input image."""
        img = self._imread(input).as_float()
        out = img.tiled(overlap=32).lowpass_filter(cutoff=cutoff, order=order)
        out.imsave(output)
        return None

    @dask_thread_worker.with_progress(desc="Binning.")
    @set_design(text="Binning")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def binning(
        self,
        input: _InputPath,
        output: _OutputPath,
        bin_size: Annotated[int, {"min": 2, "max": 16, "step": 1}] = 4,
    ):
        """Bin image."""
        img = self._imread(input)
        out = img.binning(bin_size, check_edges=False)
        out.imsave(output)
        return None

    @dask_thread_worker.with_progress(desc="Flipping image.")
    @set_design(text="Flip image")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def flip(
        self,
        input: _InputPath,
        output: _OutputPath,
        axes: Annotated[list[str], {"choices": ["x", "y", "z"], "widget_type": CheckBoxes}] = (),
    ):  # fmt: skip
        """Flip image by the given axes."""
        img = self._imread(input)
        for a in axes:
            img = img[ip.slicer(a)[::-1]]
        img.imsave(output)
        return None

    @set_design(text="Preview input image")
    def preview(self, input: _InputPath):
        """Open a preview of the input image."""
        from cylindra.widgets import CylindraMainWidget

        prev = view_image(input, self)
        main = self.find_ancestor(CylindraMainWidget)
        main._active_widgets.add(prev)
        return None

    def _imread(self, path, chunks=GVar.dask_chunk) -> ip.ImgArray | ip.LazyImgArray:
        img = ip.lazy.imread(path, chunks=chunks)
        if img.gb < self.max_gb:
            img = img.compute()
        return img
