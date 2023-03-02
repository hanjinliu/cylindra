import os
from typing import Annotated
from magicclass import (
    magicclass,
    vfield,
    MagicTemplate,
    set_options,
    set_design,
)
from magicclass.types import Path, OneOf
from magicclass.widgets import FloatRangeSlider
import numpy as np
import impy as ip
import operator
from napari.types import LayerDataTuple
from napari.layers import Image, Labels, Layer

from cylindra.utils import set_gpu

def _convert_array(arr: np.ndarray, scale: float) -> ip.ImgArray:
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

@magicclass(record=False)
class Volume(MagicTemplate):
    """A custom menu that provides useful functions for volumeric data visualization."""

    @set_options(auto_call=True)
    @set_design(text="Binning")
    def binning(self, layer: Image, bin_size: Annotated[int, {"min": 1, "max": 16}] = 2) -> LayerDataTuple:
        if layer is None:
            return None
        img = _convert_array(layer.data, layer.scale[-1])
        out = img.binning(binsize=bin_size, check_edges=False)
        translate: list[float] = []
        for k, v in img.scale.items():
            if k in ["z", "y", "x"]:
                translate.append((bin_size - 1) / 2 * v)
            else:
                translate.append(0.)
        
        return (
            out, 
            dict(scale=out.scale, 
                 translate=translate, 
                 name=layer.name + "-binning",
                 rendering=layer.rendering,
                 ), 
            "image",
        )
    
    @set_options(auto_call=True)
    @set_design(text="Gaussian filter")
    def gaussian_filter(self, layer: Image, sigma: Annotated[float, {"widget_type": "FloatSlider", "max": 5.0, "step": 0.1}] = 1.0) -> LayerDataTuple:
        """Apply Gaussian filter to an image."""
        return self._apply_method(layer, "gaussian_filter", sigma=sigma)
    
    @set_options(auto_call=True)
    @set_design(text="Threshold")
    def threshold(self, layer: Image, quantile: Annotated[float, {"widget_type": "FloatSlider", "max": 1.0, "step": 0.01}] = 0.5) -> LayerDataTuple:
        """Apply threshold to an image."""
        thr = np.quantile(layer.data, quantile)
        return self._apply_method(layer, "threshold", thr)

    @set_options(layout="horizontal", labels=False, auto_call=True)
    @set_design(text="Binary operation")
    def binary_operation(self, layer_1: Image, op: OneOf[OPERATORS], layer_2: Image) -> LayerDataTuple:
        if layer_1 is None or layer_2 is None:
            return None
        img1 = _convert_array(layer_1.data, layer_1.scale[-1])
        img2 = _convert_array(layer_2.data, layer_2.scale[-1])
        out = getattr(operator, op)(img1, img2)
        return (
            out,
            dict(scale=layer_1.scale, 
                 translate=layer_1.translate, 
                 name=f"{layer_1.name}-binary_op", 
                 rendering=layer_1.rendering,),
            "image",
        )

    @set_design(text="Save volume")
    def save_volume(self, layer: Image, path: Path.Save):
        """Save a volume as tif or mrc file."""
        img = layer.data
        if not isinstance(img, ip.ImgArray):
            raise TypeError(f"Use napari built-in menu to save {type(img)}.")
        
        fp, ext = os.path.split(path)
        if ext == ".mrc" and img.ndim not in (2, 3):
            if os.path.exists(fp):
                raise FileExistsError
            os.mkdir(fp)
            imgs: ip.ImgArray = img.reshape(-1, *img.sizesof("zyx"))
            imgs.axes = "pzyx"
            for i, img0 in enumerate(imgs):
                img0.imsave(os.path.join(fp, f"image-{i}.mrc"))
        else:
            img.imsave(path)
    
    @set_design(text="Save label as mask")
    def save_label_as_mask(self, layer: Labels, path: Path.Save):
        """Save a label as mask."""
        lbl: np.ndarray = layer.data
        if lbl.min() != 0 or np.unique(lbl).size != 2:
            raise ValueError("The label must be binary.")
        if lbl.ndim == 3:
            axes = "zyx"
        else:
            axes = None
        lbl = ip.asarray(lbl, axes=axes, dtype=np.bool_).set_scale(xyz=layer.scale[-1], unit="nm")
        lbl.imsave(path)
    
    @set_design(text="Plane clip")
    def plane_clip(self):
        """Open a plane clipper as an dock widget."""
        widget = PlaneClip()
        self.parent_viewer.window.add_dock_widget(widget, area="right")
        widget._connect_layer()
        return None

    def _apply_method(self, layer: Image, method_name: str, *args, **kwargs):
        if layer is None:
            return None
        img = _convert_array(layer.data, layer.scale[-1])
        with set_gpu():
            out = getattr(img, method_name)(*args, **kwargs)
        return (
            out,
            dict(scale=layer.scale, 
                 translate=layer.translate, 
                 name=f"{layer.name}-{method_name}",
                 rendering=layer.rendering,
                 ), 
            "image",
        )

@magicclass(record=False)
class PlaneClip(MagicTemplate):
    layer = vfield(Layer)
    x = vfield(FloatRangeSlider)
    y = vfield(FloatRangeSlider)
    z = vfield(FloatRangeSlider)

    @property
    def xmin_plane(self):
        return self.layer.experimental_clipping_planes[0]
    
    @property
    def xmax_plane(self):
        return self.layer.experimental_clipping_planes[1]
    
    @property
    def ymin_plane(self):
        return self.layer.experimental_clipping_planes[2]
    
    @property
    def ymax_plane(self):
        return self.layer.experimental_clipping_planes[3]
    
    @property
    def zmin_plane(self):
        return self.layer.experimental_clipping_planes[4]
    
    @property
    def zmax_plane(self):
        return self.layer.experimental_clipping_planes[5]
    
    @x.connect
    def _update_x(self):
        xmin, xmax = self.x
        self.xmin_plane.position = (0,)*(self.layer.ndim-1) + (xmin,)
        self.xmax_plane.position = (0,)*(self.layer.ndim-1) + (xmax,)
        return None

    @y.connect
    def _update_y(self):
        ymin, ymax = self.y
        self.ymin_plane.position = (0,)*(self.layer.ndim-2) + (ymin, 0)
        self.ymax_plane.position = (0,)*(self.layer.ndim-2) + (ymax, 0)
        return None

    @z.connect
    def _update_z(self):
        zmin, zmax = self.z
        self.zmin_plane.position = (0,)*(self.layer.ndim-3) + (zmin, 0, 0)
        self.zmax_plane.position = (0,)*(self.layer.ndim-3) + (zmax, 0, 0)
        return None
    
    @layer.connect
    def _connect_layer(self):
        layer = self.layer
        if not layer:
            return
        xmin = layer.extent.data[0, -1]
        xmax = layer.extent.data[1, -1]
        ymin = layer.extent.data[0, -2]
        ymax = layer.extent.data[1, -2]
        zmin = layer.extent.data[0, -3]
        zmax = layer.extent.data[1, -3]
        
        self["x"].range = xmin, xmax
        self["y"].range = ymin, ymax
        self["z"].range = zmin, zmax
        
        if len(self.layer.experimental_clipping_planes) == 6:
            self.x = self.xmin_plane.position[2], self.xmax_plane.position[2]
            self.x = self.ymin_plane.position[1], self.ymax_plane.position[1]
            self.x = self.zmin_plane.position[0], self.zmax_plane.position[0]
        
        else:
            self.x = xmin, xmax
            self.y = ymin, ymax
            self.z = zmin, zmax
            ndim = layer.ndim
            
            self.layer.experimental_clipping_planes = [
                {"position": (0,)*(ndim-1)+(xmin,), "normal": (0, 0, 1), "enabled": True},
                {"position": (0,)*(ndim-1)+(xmax,), "normal": (0, 0, -1), "enabled": True},
                {"position": (0,)*(ndim-2)+(ymin, 0), "normal": (0, 1, 0), "enabled": True},
                {"position": (0,)*(ndim-2)+(ymax, 0), "normal": (0, -1, 0), "enabled": True},
                {"position": (0,)*(ndim-3)+(zmin, 0, 0), "normal": (1, 0, 0), "enabled": True},
                {"position": (0,)*(ndim-3)+(zmax, 0, 0), "normal": (-1, 0, 0), "enabled": True},
            ]
