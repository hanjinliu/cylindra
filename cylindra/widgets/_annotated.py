from typing import Annotated, Any
import napari
from cylindra.types import MoleculesLayer, get_monomer_layers
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


def assert_layer(layer: Any, viewer: "napari.Viewer") -> MoleculesLayer:
    if isinstance(layer, str):
        return viewer.layers[layer]
    elif isinstance(layer, MoleculesLayer):
        return layer
    else:
        raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")


def assert_list_of_layers(layers: Any, viewer: "napari.Viewer") -> list[MoleculesLayer]:
    if len(layers) == 0:
        raise ValueError("No layer selected.")
    if isinstance(layers, (MoleculesLayer, str)):
        layers = [layers]
    layer_normed: list[MoleculesLayer] = []
    for layer in layers:
        if isinstance(layer, str):
            layer_normed.append(viewer.layers[layer])
        elif isinstance(layer, MoleculesLayer):
            layer_normed.append(layer)
        else:
            raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")
    return layer_normed
