from typing import TYPE_CHECKING, NewType

import macrokit
import magicgui
import napari
import numpy as np
from napari.layers import Layer

from cylindra._napari import MoleculesLayer
from cylindra.const import PREVIEW_LAYER_NAME

if TYPE_CHECKING:
    from magicgui.widgets.bases import CategoricalWidget


def _viewer_ancestor() -> "napari.Viewer | None":
    from cylindra.core import instance

    match instance():
        case None:
            return None
        case ui:
            return ui.parent_viewer


# This function will be called by magicgui to find all the available monomer layers.
def get_monomer_layers(gui: "CategoricalWidget") -> list[MoleculesLayer]:
    viewer = _viewer_ancestor()
    if not viewer:
        return []
    return [
        x
        for x in viewer.layers
        if isinstance(x, MoleculesLayer) and x.name != PREVIEW_LAYER_NAME
    ]


if TYPE_CHECKING:
    ColoredLayer = MoleculesLayer
else:
    ColoredLayer = NewType("ColoredLayer", Layer)


def get_colored_layers(gui: "CategoricalWidget") -> "list[ColoredLayer]":
    viewer = _viewer_ancestor()
    if not viewer:
        return []
    return [
        x
        for x in viewer.layers
        if isinstance(x, (MoleculesLayer,)) and x.name != PREVIEW_LAYER_NAME
    ]


magicgui.register_type(MoleculesLayer, choices=get_monomer_layers)
magicgui.register_type(ColoredLayer, choices=get_colored_layers)

# Record 1D numpy array as a list of floats.
macrokit.register_type(np.ndarray, lambda arr: str(arr.tolist()))
