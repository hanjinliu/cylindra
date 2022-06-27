from typing import NewType, List
import numpy as np
import impy as ip
import magicgui
from magicgui.widgets._bases import CategoricalWidget
from napari.utils._magicgui import find_viewer_ancestor
from napari.layers import Points
from .const import MOLECULES


# Define a new type to list up monomer-bound layers.

MonomerLayer = NewType("MonomerLayer", Points)

def get_monomer_layers(gui: CategoricalWidget) -> List[Points]:
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, Points) and MOLECULES in x.metadata]

magicgui.register_type(MonomerLayer, choices=get_monomer_layers)

# Record 1D numpy

from macrokit import register_type
register_type(np.ndarray, lambda arr: str(arr.tolist()))
register_type(ip.ImgArray, lambda img: f"<image from {img.source}>")
