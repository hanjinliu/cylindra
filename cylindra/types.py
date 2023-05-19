from typing import TYPE_CHECKING, NewType, Union
import re
import numpy as np
import impy as ip
import polars as pl
import magicgui
from napari.utils._magicgui import find_viewer_ancestor
from napari.layers import Layer
from cylindra._custom_layers import MoleculesLayer, CylinderLabels

if TYPE_CHECKING:
    from magicgui.widgets._bases import CategoricalWidget


# This function will be called by magicgui to find all the available monomer layers.
def get_monomer_layers(gui: "CategoricalWidget") -> list[MoleculesLayer]:
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, MoleculesLayer)]


def get_colored_layers(
    gui: "CategoricalWidget",
) -> "list[MoleculesLayer | CylinderLabels]":
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, (MoleculesLayer, CylinderLabels))]


if TYPE_CHECKING:
    ColoredLayer = Union[MoleculesLayer, CylinderLabels]
else:
    ColoredLayer = NewType("ColoredLayer", Layer)

magicgui.register_type(MoleculesLayer, choices=get_monomer_layers)
magicgui.register_type(ColoredLayer, choices=get_colored_layers)

# Record 1D numpy array as a list of floats.
from macrokit import register_type, parse

register_type(np.ndarray, lambda arr: str(arr.tolist()))
register_type(ip.ImgArray, lambda img: f"<image from {img.source}>")

_NumberPattern = r"[0-9]\.?[0-9]*"
_RustTypes = r"(i8|i16|i32|i64|u8|u16|u32|u64|f32|f64)"
_RustNumber = re.compile(f"\\(({_NumberPattern}){_RustTypes}\\)")
_RustUtf8 = re.compile("\\(Utf8\\((.*)\\)\\)")


@register_type(pl.Expr)
def format_pl_expr(expr: pl.Expr) -> str:
    """Format polars.Expr input as a evaluable string."""
    expr_str = str(expr)
    expr_str = _RustNumber.sub(r"\1", expr_str)
    expr_str = _RustUtf8.sub(r"'\1'", expr_str)
    mk_expr = parse(expr_str)
    str_expr = str(mk_expr.format({"col": "pl.col"}))
    return str_expr.replace("[(", "(").replace(")]", ")")
