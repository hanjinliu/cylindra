from __future__ import annotations
import os
from magicclass.widgets import Slider, SpreadSheet, ConsoleTextEdit
from magicclass import (
    magicclass,
    MagicTemplate,
    field,
    vfield,
)
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.ext.vispy import Vispy3DCanvas

import numpy as np
import impy as ip
import pandas as pd

@magicclass
class ImagePreview(MagicTemplate):
    canvas = field(QtImageCanvas, options={"lock_contrast_limits": True})
    sl = field(Slider, name="slice")
    @magicclass(widget_type="frame", layout="horizontal")
    class Fft(MagicTemplate):
        apply_filter = vfield(False)
        cutoff = vfield(0.5, options={"min": 0.05, "max": 0.85, "step": 0.05, "visible": False})
        @apply_filter.connect
        def _toggle(self):
            self["cutoff"].visible = self.apply_filter
    
    def _load_image(self, path: str):
        if self.canvas.image is not None:
            del self.canvas.image
        img = ip.lazy_imread(path, chunks=(1, "auto", "auto"))
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
            img_slice = img_slice.tiled_lowpass_filter(
                cutoff=self.Fft.cutoff, chunks=(496, 496)
            )
        self.canvas.image = np.asarray(img_slice)
    
    @classmethod
    def _imread(cls, path: str):
        self = cls()
        self._load_image(path)
        self.show()

        
def view_tables(paths: list[str], parent: MagicTemplate = None, **kwargs) -> SpreadSheet:
    xl = SpreadSheet()
    for i, path in enumerate(paths):
        df = pd.read_csv(path, **kwargs)
        xl.append(df)
        xl.rename(i, os.path.basename(path))
    xl.native.setParent(parent.native, xl.native.windowFlags())
    xl.show()
    return xl

def view_text(path: str, parent: MagicTemplate = None, **kwargs) -> ConsoleTextEdit:
    with open(path, mode="r", **kwargs) as f:
        txt = f.read()
    
    textedit = ConsoleTextEdit(value=txt)
    textedit.native.setParent(parent.native, textedit.native.windowFlags())
    textedit.show()
    return textedit

def view_image(path: str, parent: MagicTemplate = None) -> ImagePreview:
    prev = ImagePreview()
    prev._load_image(path)
    prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.show()
    return prev

def view_surface(data, parent: MagicTemplate = None) -> Vispy3DCanvas:
    prev = Vispy3DCanvas()
    prev.add_surface(data)
    prev.native.setParent(parent.native, prev.native.windowFlags())
    prev.show()
    return prev
