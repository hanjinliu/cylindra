from __future__ import annotations

import os
from magicclass.widgets import Slider, TabbedContainer, ConsoleTextEdit
from magicclass import (
    magicclass,
    MagicTemplate,
    field,
    vfield,
)
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.ext.polars import DataFrameView

import numpy as np
import impy as ip
import polars as pl


@magicclass(record=False)
class ImagePreview(MagicTemplate):
    """A widget to preview 3D image by 2D slices."""

    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)
    sl = field(int, widget_type=Slider, name="slice")

    @magicclass(widget_type="frame", layout="horizontal")
    class Fft(MagicTemplate):
        """
        FFT parameters.

        Attributes
        ----------
        apply_filter : bool
            Apply low-pass filter to image.
        cutoff : float
            Cutoff frequency for low-pass filter.
        """

        apply_filter = vfield(True)
        cutoff = vfield(0.05).with_options(min=0.05, max=0.85, step=0.05, visible=False)

        @apply_filter.connect
        def _toggle(self):
            self["cutoff"].visible = self.apply_filter

    def _load_image(self, path: str):
        if self.canvas.image is not None:
            del self.canvas.image
        img = ip.lazy.imread(path, chunks=(4, "auto", "auto"))
        if img.ndim != 3:
            raise ValueError("Cannot only preview 3D image.")
        self._img = img
        slmax = img.shape[0] - 1
        self.sl.value = min(slmax, self.sl.value)
        self.sl.max = slmax
        self._update_canvas()

    @sl.connect
    @Fft.apply_filter.connect
    @Fft.cutoff.connect
    def _update_canvas(self):
        img_slice = self._img[self.sl.value].compute()
        if self.Fft.apply_filter:
            img_slice = img_slice.tiled(chunks=(496, 496)).lowpass_filter(
                cutoff=self.Fft.cutoff
            )
        self.canvas.image = np.asarray(img_slice)

    @Fft.apply_filter.connect
    def _auto_contrast(self):
        min_, max_ = np.percentile(self.canvas.image, [1, 97])
        self.canvas.contrast_limits = (min_, max_)

    @classmethod
    def _imread(cls, path: str):
        self = cls()
        self._load_image(path)
        self.show()
        self.Fft.apply_filter = True


def view_tables(
    paths: list[str], parent: MagicTemplate = None, **kwargs
) -> TabbedContainer:
    """Preview a list of tables."""
    container = TabbedContainer(labels=False)
    for i, path in enumerate(paths):
        df = pl.read_csv(path, **kwargs)
        view = DataFrameView(value=df, name=os.path.basename(path))
        container.append(view)
    container.native.setParent(parent.native, container.native.windowFlags())
    container.show()
    return container


def view_text(path: str, parent: MagicTemplate = None, **kwargs) -> ConsoleTextEdit:
    """Preview a text file."""
    with open(path, mode="r", **kwargs) as f:
        txt = f.read()

    textedit = ConsoleTextEdit(value=txt)
    textedit.native.setParent(parent.native, textedit.native.windowFlags())
    textedit.show()
    return textedit


def view_image(path: str, parent: MagicTemplate = None) -> ImagePreview:
    """Preview an image."""
    prev = ImagePreview()
    prev._load_image(path)
    prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.show()
    return prev


def view_surface(data, parent: MagicTemplate = None) -> Vispy3DCanvas:
    """Preview a 3D surface."""
    prev = Vispy3DCanvas()
    prev.add_surface(data)
    prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.show()
    return prev
