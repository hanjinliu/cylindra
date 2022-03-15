from pathlib import Path
from magicclass import magicmenu, MagicTemplate, set_options
import impy as ip
from ..const import GVar

@magicmenu
class ImageProcessor(MagicTemplate):
    @set_options(dtype={"choices": ["int8", "uint8", "uint16", "float32"]})
    def Convert_dtype(self, input: Path, dtype, output: Path):
        img = ip.lazy_imread(input, chunks=GVar.daskChunk)
        out = img.as_img_type(dtype)
        out.imsave(output)
        return None
        
    def Invert(self, input: Path, output: Path):
        img = ip.lazy_imread(input)
        out = -img
        out.imsave(output)
        return None
