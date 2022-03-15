from pathlib import Path
from magicclass import magicmenu, MagicTemplate, set_options
import impy as ip
from ..const import GVar

@magicmenu
class ImageProcessor(MagicTemplate):
    @set_options(dtype={"choices": ["int8", "uint8", "uint16", "float32"]})
    def Convert_dtype(self, input: Path, dtype, output: Path):
        img = self._imread(input)
        out = img.as_img_type(dtype)
        out.imsave(output)
        return None
        
    def Invert(self, input: Path, output: Path):
        img = self._imread(input)
        out = -img
        out.imsave(output)
        return None
    
    @set_options(cutoff={"min": 0.05, "max": 0.85, "step": 0.05, "value": 0.5})
    def Lowpass_filter(self, input: Path, cutoff: float, output: Path):
        img = self._imread(input)
        out = img.tiled_lowpass_filter(cutoff, update=True, overlap=32)
        out.imsave(output)
        return None

    def _imread(self, path):
        return ip.lazy_imread(path, chunks=GVar.daskChunk)
