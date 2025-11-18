from glob import glob
from pathlib import Path
from typing import Annotated, Literal

import impy as ip
from magicclass import confirm, magicclass, set_design, vfield
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


_TOTAL = "self._file_count(input)"


@magicclass(record=False)
class ImageProcessor(ChildWidget):
    """Process image files.

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

    @dask_thread_worker.with_progress(desc="Converting data type ...", total=_TOTAL)
    @set_design(text="Convert dtype")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def convert_dtype(
        self,
        input: _InputPath,
        output: _OutputPath,
        dtype: Literal["int8", "uint8", "uint16", "float16", "float32"],
    ):
        """Convert data type of the input image."""
        img = self._imread(input)
        out = img.as_img_type(dtype)
        yield from self._imsave(out, output)

    @dask_thread_worker.with_progress(desc="Inverting image ...", total=_TOTAL)
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

    @dask_thread_worker.with_progress(desc="Changing pixel size ...", total=_TOTAL)
    @set_design(text="Change pixel size")
    @confirm(text="Output path alreadly exists, overwrite?", condition=_confirm_path)
    def change_pixel_size(
        self,
        input: _InputPath,
        output: _OutputPath,
        scale: Annotated[float, {"min": 0.01, "max": 10.0, "step": 0.0001}] = 1.0,
    ):
        """Change pixel size of the input image."""
        img = self._imread(input)
        if isinstance(img, ip.DataList):
            out = ip.DataList([each.set_scale(zyx=scale, unit="nm") for each in img])
        else:
            out = img.set_scale(zyx=scale, unit="nm")
        yield from self._imsave(out, output)

    @dask_thread_worker.with_progress(desc="Low-pass filtering ...", total=_TOTAL)
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

    @dask_thread_worker.with_progress(desc="Binning ...", total=_TOTAL)
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

    @dask_thread_worker.with_progress(desc="Flipping image.", total=_TOTAL)
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

    @set_design(text="Preview input image")
    def preview(self, input: _InputPath):
        """Open a preview of the input image."""
        _input = str(input)
        if _is_wildcard(_input):
            input_normed = glob(_input, recursive=True)
        else:
            input_normed = _input
        prev = view_image(input_normed, self)
        ACTIVE_WIDGETS.add(prev)

    def _imread(self, path) -> "ip.LazyImgArray | ip.DataList[ip.LazyImgArray]":
        path = str(path)
        self._current_suffix = self.suffix
        if _is_wildcard(path):
            if self.suffix is None:
                raise ValueError("Cannot read multiple images without `suffix`.")
            imgs: list[ip.LazyImgArray] = []
            for fp in glob(path, recursive=True):
                imgs.append(ip.lazy.imread(fp, chunks=(4, -1, -1)))
            out = ip.DataList(imgs)
        else:
            out = ip.lazy.imread(path, chunks=(4, -1, -1))
        return out

    def _file_count(self, path) -> int:
        # NOTE: this is used in _TOTAL for progress bar
        path = str(path)
        if _is_wildcard(path):
            if self.suffix is None:
                return 0
            return len(glob(path, recursive=True))
        else:
            return 1

    def _imsave(self, img: "ip.LazyImgArray | ip.DataList[ip.LazyImgArray]", path):
        if isinstance(img, ip.DataList):
            for each in img:
                save_path = _autofill(each.source, self._current_suffix)
                yield _imsave_impl(each, save_path)
        else:
            yield _imsave_impl(img, path)


def _imsave_impl(img: "ip.LazyImgArray", save_path: Path, thresh: int = 5e7):
    if img.size < thresh:
        img = img.compute()
    is_overwrite = save_path.exists()
    if is_overwrite:
        save_path_temp = save_path.with_stem(save_path.stem + "-tmp")
        i = 0
        while save_path_temp.exists():
            save_path.with_stem(save_path.stem + f"-tmp{i}")
        bak_path = save_path.with_name(save_path.name + "~")
        if bak_path.exists():
            bak_path.unlink()
        img.imsave(save_path_temp)  # data -> save_path_temp
        save_path.rename(bak_path)  # save_path -> bak_path
        save_path_temp.rename(save_path)  # save_path_temp -> save_path
    else:
        img.imsave(save_path)


def _is_wildcard(path_str: str) -> bool:
    return "*" in path_str or "?" in path_str
