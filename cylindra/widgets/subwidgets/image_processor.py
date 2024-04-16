from glob import glob
from pathlib import Path
from typing import Annotated, Literal

import impy as ip
from magicclass import (
    confirm,
    magicclass,
    set_design,
    vfield,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.types import Optional

from cylindra._previews import view_image
from cylindra.const import FileFilter
from cylindra.core import ACTIVE_WIDGETS
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.widgets.subwidgets._child_widget import ChildWidget


def _autofill(input_path, suffix: str) -> Path:
    input_path = Path(input_path)
    output_path = input_path.with_stem(input_path.stem + suffix)

    n = 0
    while output_path.exists():
        output_path = output_path.with_stem(output_path.stem + f"-{n}")
    return output_path


@magicclass(record=False)
class ImageProcessor(ChildWidget):
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
    suffix = vfield(Optional[str]).with_options(value="-output", text="Do not autofill")
    output_image = vfield(Path).with_options(filter=FileFilter.IMAGE, mode="w")

    _InputPath = Annotated[Path, {"bind": input_image, "widget_type": "EmptyWidget"}]
    _OutputPath = Annotated[Path, {"bind": output_image, "widget_type": "EmptyWidget"}]

    def __init__(self):
        self._current_suffix = "-output"

    @input_image.connect
    @suffix.connect
    def _autofill_output_path(self):
        if self.suffix is None:
            return
        self.output_image = _autofill(self.input_image, self.suffix)

    def _confirm_path(self):
        return self.output_image.exists()

    @dask_thread_worker.with_progress(
        desc="Converting data type.", total="self._file_count(input)"
    )
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
        yield from self._imsave(out, output)
        return None

    @dask_thread_worker.with_progress(
        desc="Inverting image.", total="self._file_count(input)"
    )
    @set_design(text="Invert intensity")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def invert(
        self,
        input: _InputPath,
        output: _OutputPath,
    ):
        """Invert intensity of the input image."""
        img = self._imread(input)
        if isinstance(img, ip.DataList):
            out = ip.DataList([-each for each in img])
        else:
            out = -img
        yield from self._imsave(out, output)
        return None

    @dask_thread_worker.with_progress(
        desc="Low-pass filtering.", total="self._file_count(input)"
    )
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
        yield from self._imsave(out, output)
        return None

    @dask_thread_worker.with_progress(desc="Binning.", total="self._file_count(input)")
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
        yield from self._imsave(out, output)
        return None

    @dask_thread_worker.with_progress(
        desc="Flipping image.", total="self._file_count(input)"
    )
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
        yield from self._imsave(img, output)
        return None

    @set_design(text="Preview input image")
    def preview(self, input: _InputPath):
        """Open a preview of the input image."""
        if "*" in str(input):
            input = glob(str(input), recursive=True)
        prev = view_image(input, self)
        ACTIVE_WIDGETS.add(prev)
        return None

    def _imread(self, path) -> "ip.LazyImgArray | ip.DataList[ip.LazyImgArray]":
        path = str(path)
        self._current_suffix = self.suffix
        if "*" in path:
            if self.suffix is None:
                raise ValueError("Cannot read multiple images without `suffix`.")
            imgs = []
            for fp in glob(path, recursive=True):
                imgs.append(ip.lazy.imread(fp, chunks=(4, -1, -1)))
            out = ip.DataList(imgs)
        else:
            out = ip.lazy.imread(path, chunks=(4, -1, -1))
        return out

    def _file_count(self, path) -> int:
        path = str(path)
        if "*" in path:
            if self.suffix is None:
                return 0
            return len(glob(path, recursive=True))
        else:
            return 1

    def _imsave(
        self, img: "ip.LazyImgArray | ip.DataList[ip.LazyImgArray]", path: Path
    ):
        if isinstance(img, ip.DataList):
            for each in img:
                save_path = _autofill(each.source, self._current_suffix)
                if each.size < 5e7:
                    each = each.compute()
                each.imsave(save_path)
                yield
        else:
            if img.size < 5e7:
                img = img.compute()
            img.imsave(path)
            yield
        return None
