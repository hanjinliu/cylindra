import os
from pathlib import Path
from magicgui.widgets import ProgressBar
from magicclass import (
    magicclass,
    MagicTemplate,
    set_options,
    vfield,
    do_not_record,
    set_design,
)
from magicclass.utils import thread_worker
import impy as ip
from .widget_utils import FileFilter
from ._previews import view_image
from ..const import GVar


@magicclass
class ImageProcessor(MagicTemplate):
    """
    Process image files.
    
    Attributes
    ----------
    input_image : Path
        Path to the input image file.
    output_image : Path
        Path to the output image file.
    """
    input_image = vfield(Path, options={"filter": FileFilter.IMAGE}, record=False)
    output_image = vfield(Path, options={"filter": FileFilter.IMAGE, "mode": "w"}, record=False)
    pbar = vfield(ProgressBar, options={"visible": False}, record=False)
    
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
    @do_not_record
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
        order={"max": 20},
    )
    @do_not_record
    def lowpass_filter(self, cutoff: float, order: int = 2):
        """Apply Butterworth's low-pass filter to the input image."""
        img = self._imread(self.input_image)
        out = img.tiled_lowpass_filter(cutoff, overlap=32, order=order)
        out.imsave(self.output_image)
        return None
    
    @set_design(text="Preview input image.")
    @do_not_record
    def preview(self):
        """Open a preview of the input image."""
        view_image(self.input_image, self)
        return None
        
    def _imread(self, path, chunks=GVar.daskChunk):
        return ip.lazy_imread(path, chunks=chunks)
