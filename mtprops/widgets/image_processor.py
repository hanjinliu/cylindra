import os
from pathlib import Path
from magicgui.widgets import ProgressBar, Slider
from magicclass import (
    magicclass,
    MagicTemplate,
    set_options,
    field,
    vfield,
    do_not_record,
    set_design,
)
from magicclass.qthreading import thread_worker
from magicclass.ext.pyqtgraph import QtImageCanvas
import numpy as np
import impy as ip
from .widget_utils import FileFilter
from ..const import GVar


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
        with ip.silent():
            img_slice = self._img[self.sl.value].compute()
            if self.Fft.apply_filter:
                img_slice = img_slice.tiled_lowpass_filter(
                    cutoff=self.Fft.cutoff, chunks=(496, 496)
                )
        self.canvas.image = np.asarray(img_slice)
    
    
@magicclass
class ImageProcessor(MagicTemplate):
    input_image = vfield(Path, options={"filter": FileFilter.IMAGE}, record=False)
    output_image = vfield(Path, options={"filter": FileFilter.IMAGE, "mode": "w"}, record=False)
    pbar = vfield(ProgressBar, options={"visible": False}, record=False)
    
    _preview = field(ImagePreview)

    @input_image.connect
    def _generate_output_path(self):
        input_path, ext = os.path.splitext(self.input_image)
        n = 0
        while os.path.exists(f"{input_path}-{n}{ext}"):
            n += 1
        self.output_image = f"{input_path}-{n}{ext}"
    
    @thread_worker(progress={"desc": "Converting data type."})
    @set_options(dtype={"choices": ["int8", "uint8", "uint16", "float32"]})
    @set_design(text="Convert dtype")
    @do_not_record
    def convert_dtype(self, dtype):
        """Convert data type of the input image."""
        img = self._imread(self.input_image)
        out = img.as_img_type(dtype)
        out.imsave(self.output_image)
        return None
    
    @thread_worker(progress={"desc": "Inverting image."})
    @set_design(text="Invert")
    def invert(self):
        """Invert intensity of the input image."""
        img = self._imread(self.input_image)
        out = -img
        out.imsave(self.output_image)
        return None
    
    @thread_worker(progress={"desc": "Low-pass filtering."})
    @set_design(text="Low-pass filter")
    @set_options(
        cutoff={"min": 0.05, "max": 0.85, "step": 0.05, "value": 0.5},
        order={"max": 20,}
    )
    def lowpass_filter(self, cutoff: float, order: int = 2):
        """Apply Butterworth's low-pass filter to the input image."""
        img = self._imread(self.input_image)
        out = img.tiled_lowpass_filter(cutoff, overlap=32, order=order)
        with ip.silent():
            out.imsave(self.output_image)
        return None
    
    @set_design(text="Preview")
    def preview(self):
        """Open a preview of the input image."""
        self._preview.show()
        self._preview._load_image(self.input_image)

    def _imread(self, path, chunks=GVar.daskChunk):
        return ip.lazy_imread(path, chunks=chunks)
