from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING

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

from cylindra._io import lazy_imread
from cylindra.core import ACTIVE_WIDGETS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cylindra.components import CylSpline
    from cylindra.components.tomogram._tilt_series import TiltSeriesSource
    from cylindra.widgets.main import CylindraMainWidget


@magicclass(record=False, name="Image Preview", use_native_menubar=False)
class ImagePreview(MagicTemplate):
    """A widget to preview 3D image by 2D slices."""

    @magicmenu
    class Menu(MagicTemplate):
        close_this = abstractapi()

    def _get_image_choices(self, w=None) -> list[tuple[str, ip.LazyImgArray]]:
        lst: list[tuple[str, ip.LazyImgArray]] = []
        for path in self._path_choices:
            img = lazy_imread(path, chunks=(4, "auto", "auto"))
            if img.ndim != 3:
                raise ValueError("Can only preview 3D image.")
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
        self._path_choices = []

    @magicclass(widget_type="frame", layout="horizontal")
    class Filter(MagicTemplate):
        """Filtering parameters.

        Attributes
        ----------
        bin_size : int
            Image binning prior to low-pass filtering.
        apply_filter : bool
            Apply low-pass filter to image.
        cutoff : float
            Cutoff frequency for low-pass filtering.
        """

        bin_size = vfield(2).with_options(min=1, max=10, step=1)
        apply_filter = vfield(False, label="Apply low-pass filter")
        cutoff = vfield(0.1).with_options(min=0.05, max=0.85, step=0.05, enabled=False)

        @apply_filter.connect
        def _toggle(self):
            self["cutoff"].enabled = self.apply_filter

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


_SPLINE_COLORS = ["orange", "cyan", "magenta", "yellow", "lime"]


@magicclass(record=False, name="Tilt Series Viewer", use_native_menubar=False)
class TiltSeriesViewer(MagicTemplate):
    """A widget to view the tilt images with the splines projected on them.

    The splines are projected by the current alignment of the tilt series, so the
    alignment can be checked by whether the splines follow the filaments in all the
    tilts. The edges of the cylinders (dotted lines) are also shown if the radii of
    the splines are measured. The raw tilt images are shown (not preprocessed), with
    the contrast adjusted for each tilt.
    """

    @magicmenu
    class Menu(MagicTemplate):
        refresh = abstractapi()
        close_this = abstractapi()

    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)
    sl = field(int, widget_type=Slider, name="tilt")

    @magicclass(widget_type="frame", layout="horizontal")
    class Filter(MagicTemplate):
        """Filtering parameters.

        Attributes
        ----------
        bin_size : int
            Image binning prior to low-pass filtering.
        apply_filter : bool
            Apply low-pass filter to image.
        cutoff : float
            Cutoff frequency for low-pass filtering.
        """

        bin_size = vfield(4).with_options(min=1, max=16, step=1)
        apply_filter = vfield(False, label="Apply low-pass filter")
        cutoff = vfield(0.1).with_options(min=0.05, max=0.85, step=0.05, enabled=False)

        @apply_filter.connect
        def _toggle(self):
            self["cutoff"].enabled = self.apply_filter

    show_splines = vfield(True, label="Show splines")

    def __init__(self):
        self._main_ref = lambda: None

    def __post_init__(self):
        self.canvas.text_overlay.color = "lime"
        self.canvas.text_overlay.visible = True

    @set_design(text="Refresh", location=Menu)
    def refresh(self):
        """Read the tilt series and the splines again, such as after alignment."""
        self._update_canvas()

    @set_design(text="Close", location=Menu)
    def close_this(self):
        self.close()

    def _set_main(self, main: CylindraMainWidget):
        self._main_ref = weakref.ref(main)
        angles = self._get_source().tilt_series.tilt_angles
        self.sl.max = angles.size - 1
        self.sl.value = int(np.argmin(np.abs(angles)))  # the zero tilt

    def _get_main(self) -> CylindraMainWidget:
        if (main := self._main_ref()) is None:
            raise ValueError("The main widget is not available.")
        return main

    def _get_source(self) -> TiltSeriesSource:
        if (src := self._get_main().tomogram.tilt_series) is None:
            raise ValueError("The tomogram is not opened from a tilt series.")
        return src

    @show_splines.connect_async(timeout=0.1)
    @Filter.bin_size.connect_async(timeout=0.1)
    @Filter.apply_filter.connect_async(timeout=0.1)
    @Filter.cutoff.connect_async(timeout=0.1)
    @sl.connect_async(timeout=0.1, abort_limit=0.1)
    def _update_canvas(self):
        src = self._get_source()
        angles = src.tilt_series.tilt_angles
        t = min(self.sl.value, angles.size - 1)
        binsize = self.Filter.bin_size
        yield self._set_text_overlay.with_args("reading ...")
        img = ip.asarray(src.tilt_image_source()[t], axes="yx")
        if binsize > 1:
            img = img.binning(binsize, check_edges=False)
        if self.Filter.apply_filter:
            img = img.lowpass_filter(cutoff=self.Filter.cutoff)
        if self.show_splines:
            splines = list(self._get_main().tomogram.splines)
            curves = [_projected_spline(src, spl, t, binsize) for spl in splines]
        else:
            curves = []
        text = f"tilt {t}: {angles[t]:.1f}\N{DEGREE SIGN}"
        yield self._set_canvas.with_args(np.asarray(img), curves, text)

    @thread_worker.callback
    def _set_text_overlay(self, txt: str):
        self.canvas.text_overlay.text = txt

    @thread_worker.callback
    def _set_canvas(
        self,
        img: NDArray[np.float32],
        curves: list[tuple[NDArray[np.float64], list[NDArray[np.float64]]]],
        text: str,
    ):
        self.canvas.image = img
        self.canvas.contrast_limits = np.percentile(img, (0.5, 99.5))
        self.canvas.layers.clear()
        for i, (center, edges) in enumerate(curves):
            color = _SPLINE_COLORS[i % len(_SPLINE_COLORS)]
            self.canvas.add_curve(center[:, 1], center[:, 0], color=color, lw=1.5)
            for edge in edges:
                self.canvas.add_curve(edge[:, 1], edge[:, 0], color=color, ls=":")
            self.canvas.add_text(center[0, 1], center[0, 0], str(i), color=color)
        self.canvas.text_overlay.text = text


def _projected_spline(
    src: TiltSeriesSource, spl: CylSpline, t: int, binsize: int
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """The spline and the edges of its cylinder projected to the binned t-th tilt.

    Returned arrays are (y, x) positions (N, 2) in pixel of the binned image, sampled
    every ~2 nm along the spline. The edges are empty if the radius is not measured.
    """
    coords = spl.partition(max(int(spl.length() / 2), 2))
    # the binned pixel i is centered at the raw pixel i * binsize + (binsize - 1) / 2
    center = (src.project_points(coords)[:, t] - (binsize - 1) / 2) / binsize
    if spl.radius is None:
        return center, []
    tangent = np.gradient(center, axis=0)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    # the projection of a cylinder is as wide as its diameter
    radius = spl.radius / src.scale / binsize
    return center, [center + radius * normal, center - radius * normal]


def view_tilt_series(main: CylindraMainWidget) -> TiltSeriesViewer:
    """View the tilt images of the tomogram with the splines projected on them."""
    viewer = TiltSeriesViewer()
    viewer._set_main(main)
    viewer.native.setParent(main.native, viewer.native.windowFlags())
    viewer.show()
    viewer.width, viewer.height = 400, 520
    ACTIVE_WIDGETS.add(viewer)
    viewer._update_canvas()
    return viewer


def view_tables(
    paths: str | list[str], parent: MagicTemplate = None, **kwargs
) -> TabbedContainer:
    """Preview a list of tables."""
    if isinstance(paths, (str, Path)):
        df = pl.read_csv(paths, **kwargs)
        container = DataFrameView(value=df, name=Path(paths).name)
    else:
        container = TabbedContainer(labels=False)
        for path in paths:
            df = pl.read_csv(path, **kwargs)
            view = DataFrameView(value=df, name=Path(path).name)
            container.append(view)
    if parent is not None:
        container.native.setParent(parent.native, container.native.windowFlags())
    container.show()
    ACTIVE_WIDGETS.add(container)
    return container


def view_image(
    path: str | list[str],
    parent: MagicTemplate = None,
    default_bin_size: int = 2,
) -> ImagePreview:
    """Preview an image."""
    prev = ImagePreview()
    if isinstance(path, list):
        prev._load_images(path)
    else:
        prev._load_image(path)
    if parent is not None:
        prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.Filter.bin_size = default_bin_size
    prev.show()
    prev.width, prev.height = 320, 440
    ACTIVE_WIDGETS.add(prev)
    return prev
