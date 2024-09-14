from typing import Annotated, Any, Literal

import napari
from magicclass._gui import BaseGui
from magicclass.types import Optional

from cylindra.types import MoleculesLayer, get_monomer_layers, get_splines
from cylindra.widgets._widget_ext import CheckBoxes


def _as_layer_name(self: Any, layer: MoleculesLayer | str) -> str:
    if isinstance(layer, str):
        return layer
    elif isinstance(layer, MoleculesLayer):
        return layer.name
    raise TypeError(f"Expected str or MoleculesLayer, got {type(layer)}")


def _as_layer_names(self: Any, layers: list[MoleculesLayer | str]) -> str:
    out = []
    if isinstance(layers, (str, MoleculesLayer)):
        layers = [layers]
    for layer in layers:
        if isinstance(layer, str):
            out.append(layer)
        else:
            out.append(layer.name)
    return out


def _splines_validator(self: BaseGui, splines) -> list[int] | Literal["all"]:
    """Validate list input, or 'all' if all splines are selected."""
    from cylindra.widgets.main import CylindraMainWidget

    parent = self
    while parent is not None:
        if isinstance(parent, CylindraMainWidget):
            nspl = parent.splines.count()
            if splines is None:
                splines = list(range(nspl))
            elif isinstance(splines, str):
                if splines == "all":
                    return splines
                raise TypeError("Only 'all' is allow for a string input")
            elif splines is all:
                return "all"
            elif not hasattr(splines, "__iter__"):
                splines = [int(splines)]
            else:
                for i in splines:
                    if i >= nspl:
                        raise ValueError(f"Spline index {i} is out of range.")
                splines = sorted(splines)
            if len(splines) == 0:
                raise ValueError("No spline is selected.")
            if splines == list(range(nspl)):
                # For better reusabiligy, recording as 'all' is better.
                return "all"
            return splines
        parent = parent.__magicclass_parent__
    return splines


MoleculesLayerType = Annotated[
    MoleculesLayer,
    {"validator": _as_layer_name},
]

MoleculesLayersType = Annotated[
    list[MoleculesLayer],
    {
        "choices": get_monomer_layers,
        "widget_type": CheckBoxes,
        "value": (),
        "validator": _as_layer_names,
    },
]

FSCFreq = Annotated[
    Optional[float],
    {
        "label": "Frequency precision",
        "text": "Choose proper value",
        "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02},
    },
]

SplinesType = Annotated[
    list[int],
    {
        "choices": get_splines,
        "widget_type": CheckBoxes,
        "validator": _splines_validator,
    },
]


def assert_layer(layer: Any, viewer: "napari.Viewer") -> MoleculesLayer:
    if isinstance(layer, str):
        layer = viewer.layers[layer]
    if isinstance(layer, MoleculesLayer):
        return layer
    else:
        raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")


def assert_list_of_layers(layers: Any, viewer: "napari.Viewer") -> list[MoleculesLayer]:
    if isinstance(layers, (MoleculesLayer, str)):
        layers = [layers]
    if len(layers) == 0:
        raise ValueError("No layer selected.")
    elif not hasattr(layers, "__iter__"):
        raise TypeError(f"Expected iterable, got {type(layers)}")
    layer_normed: list[MoleculesLayer] = []
    for layer in layers:
        if isinstance(layer, str):
            layer = viewer.layers[layer]
        if isinstance(layer, MoleculesLayer):
            layer_normed.append(layer)
        else:
            raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")
    return layer_normed
