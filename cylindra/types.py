from typing import TYPE_CHECKING, NewType
import re
import numpy as np
import magicgui
from napari.utils._magicgui import find_viewer_ancestor
from napari.layers import Layer
from cylindra._custom_layers import MoleculesLayer, CylinderLabels
from cylindra.const import PREVIEW_LAYER_NAME

if TYPE_CHECKING:
    from magicgui.widgets._bases import CategoricalWidget


# This function will be called by magicgui to find all the available monomer layers.
def get_monomer_layers(gui: "CategoricalWidget") -> list[MoleculesLayer]:
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [
        x
        for x in viewer.layers
        if isinstance(x, MoleculesLayer) and x.name != PREVIEW_LAYER_NAME
    ]


def get_colored_layers(
    gui: "CategoricalWidget",
) -> "list[MoleculesLayer | CylinderLabels]":
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [
        x
        for x in viewer.layers
        if isinstance(x, (MoleculesLayer, CylinderLabels))
        and x.name != PREVIEW_LAYER_NAME
    ]


if TYPE_CHECKING:
    ColoredLayer = MoleculesLayer | CylinderLabels
else:
    ColoredLayer = NewType("ColoredLayer", Layer)

magicgui.register_type(MoleculesLayer, choices=get_monomer_layers)
magicgui.register_type(ColoredLayer, choices=get_colored_layers)

# Record 1D numpy array as a list of floats.
from macrokit import register_type, parse

register_type(np.ndarray, lambda arr: str(arr.tolist()))
