from pathlib import Path
from typing import List
from magicclass import magicmenu, MagicTemplate, set_options, do_not_record
import numpy as np
import impy as ip
import operator
from napari.types import LayerDataTuple
from napari.layers import Image

from mtprops.utils import set_gpu

def _convert_array(arr, scale: float) -> ip.ImgArray:
    if not isinstance(arr, ip.ImgArray):
        if len(arr.shape) == 3:
            arr = ip.asarray(arr, axes="zyx")
        else:
            arr = ip.asarray(arr)
        arr.set_scale(xyz=scale)
    return arr

OPERATORS = [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "truediv"),
    ("==", "eq"),
    (">=", "ge"),
    (">", "gt"),
    ("<=", "le"),
    ("<", "lt"),
]

@magicmenu
class Volume(MagicTemplate):
    """A custom menu that provides useful functions for volumeric data visualization."""

    @set_options(bin_size={"min": 1, "max": 16}, auto_call=True)
    @do_not_record
    def Binning(self, layer: Image, bin_size: int = 2) -> LayerDataTuple:
        if layer is None:
            return None
        img = _convert_array(layer.data, layer.scale[-1])
        with ip.silent():
            out = img.binning(binsize=bin_size, check_edges=False)
        translate: List[float] = []
        for k, v in img.scale.items():
            if k in "zyx":
                translate.append((bin_size - 1) / 2 * v)
            else:
                translate.append(0.)
        
        return (
            out, 
            dict(scale=out.scale, 
                 translate=translate, 
                 name=layer.name + "-binning"), 
            "image",
        )
    
    @set_options(sigma={"widget_type": "FloatSlider", "max": 5.0, "step": 0.1}, auto_call=True)
    @do_not_record
    def Gaussian_filter(self, layer: Image, sigma: float = 1.0) -> LayerDataTuple:
        return self._apply_method(layer, "gaussian_filter", sigma=sigma)
    
    @set_options(quantile={"widget_type": "FloatSlider", "max": 1.0, "step": 0.01}, auto_call=True)
    @do_not_record
    def Threshold(self, layer: Image, quantile: float = 0.5) -> LayerDataTuple:
        thr = np.quantile(layer.data, quantile)
        return self._apply_method(layer, "threshold", thr)

    @set_options(op={"choices": OPERATORS}, layout="horizontal", labels=False, auto_call=True)
    @do_not_record
    def Binary_operation(self, layer_1: Image, op, layer_2: Image) -> LayerDataTuple:
        if layer_1 is None or layer_2 is None:
            return None
        img1 = _convert_array(layer_1.data, layer_1.scale[-1])
        img2 = _convert_array(layer_2.data, layer_2.scale[-1])
        out = getattr(operator, op)(img1, img2)
        return (
            out,
            dict(scale=layer_1.scale, 
                 translate=layer_1.translate, 
                 name=f"{layer_1.name}-binary_op"), 
            "image",
        )
    
    @set_options(path={"mode": "w"})
    @do_not_record
    def Save_volume(self, layer: Image, path: Path):
        img = layer.data
        if not isinstance(img, ip.ImgArray):
            raise TypeError
        img.imsave(path)

    def _apply_method(self, layer: Image, method_name: str, *args, **kwargs):
        if layer is None:
            return None
        img = _convert_array(layer.data, layer.scale[-1])
        with ip.silent(), set_gpu():
            out = getattr(img, method_name)(*args, **kwargs)
        return (
            out,
            dict(scale=layer.scale, 
                 translate=layer.translate, 
                 name=f"{layer.name}-{method_name}"), 
            "image",
        )
