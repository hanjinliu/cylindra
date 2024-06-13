from __future__ import annotations

from pathlib import Path

import impy as ip
import numpy as np
import polars as pl
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    magicclass,
    magicmenu,
    set_design,
    vfield,
)
from magicclass.ext.polars import DataFrameView
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.utils import thread_worker
from magicclass.widgets import TabbedContainer
from magicgui.widgets import Slider

from cylindra.core import ACTIVE_WIDGETS


@magicclass(record=False, name="Image Preview", use_native_menubar=False)
class ImagePreview(MagicTemplate):
    """A widget to preview 3D image by 2D slices."""

    @magicmenu
    class Menu(MagicTemplate):
        close_this = abstractapi()

    def _get_image_choices(self, w=None) -> list[tuple[str, ip.LazyImgArray]]:
        lst = []
        for path in self._path_choices:
            img = ip.lazy.imread(path, chunks=(4, "auto", "auto"))
            if img.ndim != 3:
                raise ValueError("Cannot only preview 3D image.")
            lst.append((Path(path).as_posix(), img))
        return lst

    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)
    sl = field(int, widget_type=Slider, name="z")
    image = field().with_choices(_get_image_choices)

    def __init__(self):
        self._path_choices: list[str] = []

    def __post_init__(self):
        self.canvas.text_overlay.color = "lime"
        self.canvas.text_overlay.visible = True

    @set_design(text="Close", location=Menu)
    def close_this(self):
        self.close()

    @magicclass(widget_type="frame", layout="horizontal")
    class Filter(MagicTemplate):
        """
        Filtering parameters.

        Attributes
        ----------
        bin_size : int
            Image binning prior to low-pass filtering.
        apply_filter : bool
            Apply low-pass filter to image.
        cutoff : float
            Cutoff frequency for low-pass filtering.
        """

        bin_size = vfield(1).with_options(min=1, max=10, step=1)
        apply_filter = vfield(False)
        cutoff = vfield(0.05).with_options(min=0.05, max=0.85, step=0.05, enabled=False)

        @apply_filter.connect
        def _toggle(self):
            self["cutoff"].enabled = self.apply_filter

    def _lazy_imread(self, path: str):
        img = ip.lazy.imread(path, chunks=(1, "auto", "auto"))
        if img.ndim != 3:
            raise ValueError("Cannot only preview 3D image.")
        return Path(path).as_posix(), img

    def _load_image(self, path: str):
        if self.canvas.image is not None:
            del self.canvas.image
        self._path_choices = [path]
        self.image.reset_choices()

    def _load_images(self, paths: list[str]):
        if self.canvas.image is not None:
            del self.canvas.image
        self._path_choices = paths
        self.image.reset_choices()

    @image.connect_async(timeout=0.1)
    def _image_loaded(self, img: np.ndarray):
        if img is None:
            return
        yield from self._update_slider.arun(img)
        yield from self._update_canvas_and_auto_contrast.arun()

    @thread_worker.callback
    def _update_slider(self, img: np.ndarray):
        slmax = img.shape[0] - 1
        self.sl.value = slmax // 2
        self.sl.max = slmax

    @Filter.cutoff.connect_async(timeout=0.1)
    @sl.connect_async(timeout=0.1, abort_limit=0.1)
    def _update_canvas(self):
        yield self._set_text_overlay.with_args("reading ...")
        im = self.image.value[self.sl.value]
        if self.Filter.bin_size > 1:
            im = im.binning(self.Filter.bin_size, check_edges=False)
        if self.Filter.apply_filter:
            if im.size < 4e4:
                img_slice = im.compute().lowpass_filter(cutoff=self.Filter.cutoff)
            else:
                overlap = (min(32, s) for s in im.shape)
                img_slice = (
                    im.tiled(overlap=overlap)
                    .lowpass_filter(cutoff=self.Filter.cutoff)
                    .compute()
                )
        else:
            img_slice = im.compute()
        yield self._set_canvas_image.with_args(np.asarray(img_slice))
        yield self._set_text_overlay.with_args("")

    @thread_worker.callback
    def _set_text_overlay(self, txt: str):
        self.canvas.text_overlay.text = txt

    @thread_worker.callback
    def _set_canvas_image(self, img):
        self.canvas.image = img

    @Filter.bin_size.connect_async(timeout=0.1)
    @Filter.apply_filter.connect_async(timeout=0.1)
    def _update_canvas_and_auto_contrast(self):
        yield from self._update_canvas.arun()
        img = self.canvas.image
        if img is None:
            return
        self.canvas.contrast_limits = np.percentile(img, (0.5, 99.5))


def view_tables(
    paths: str | list[str], parent: MagicTemplate = None, **kwargs
) -> TabbedContainer:
    """Preview a list of tables."""
    if isinstance(paths, (str, Path)):
        df = pl.read_csv(paths, **kwargs)
        container = DataFrameView(value=df, name=Path(paths).name)
    else:
        container = TabbedContainer(labels=False)
        for _i, path in enumerate(paths):
            df = pl.read_csv(path, **kwargs)
            view = DataFrameView(value=df, name=Path(path).name)
            container.append(view)
    if parent is not None:
        container.native.setParent(parent.native, container.native.windowFlags())
    container.show()
    ACTIVE_WIDGETS.add(container)
    return container


def view_image(path: str | list[str], parent: MagicTemplate = None) -> ImagePreview:
    """Preview an image."""
    prev = ImagePreview()
    if isinstance(path, list):
        prev._load_images(path)
    else:
        prev._load_image(path)
    if parent is not None:
        prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.show()
    prev.width, prev.height = 320, 440
    ACTIVE_WIDGETS.add(prev)
    return prev
