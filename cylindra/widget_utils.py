from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from timeit import default_timer
from typing import TYPE_CHECKING, Annotated, Any, Sequence, TypedDict

import macrokit as mk
import napari
import numpy as np
import polars as pl
from acryo import Molecules
from magicclass.logging import getLogger
from magicclass.types import ExprStr
from magicclass.widgets import EvalLineEdit
from numpy.typing import NDArray

from cylindra.components._base import BaseComponent
from cylindra.const import nm
from cylindra.types import MoleculesLayer

if TYPE_CHECKING:
    from magicclass import MagicTemplate

    from cylindra.components import CylSpline


# namespace used in predicate
POLARS_NAMESPACE = {
    "pl": pl,
    "col": pl.col,
    "when": pl.when,
    "format": pl.format,
    "int": int,
    "float": float,
    "str": str,
    "np": np,
    "__builtins__": {},
}


def _validate_expr_or_scalar(expr: str | pl.Expr | int | float) -> str | int | float:
    if isinstance(expr, (str, pl.Expr)):
        if isinstance(expr, str):
            value = ExprStr(expr, POLARS_NAMESPACE).eval()
            if not isinstance(value, (int, float, np.number, pl.Expr)):
                raise TypeError(f"Invalid type: {type(value)}")
            return expr
        elif isinstance(expr, pl.Expr):
            return _polars_expr_to_str(expr)
        else:
            raise TypeError("Input must be string or polars.Expr type.")
    elif isinstance(expr, (int, float, np.number)):
        return expr
    else:
        raise TypeError(f"Got invalid type: {type(expr)}")


PolarsExprStrOrScalar = Annotated[
    str | int | float,
    {
        "widget_type": EvalLineEdit,
        "namespace": POLARS_NAMESPACE,
        "validator": _validate_expr_or_scalar,
    },
]


def _unwrap_rust_expr(expr: mk.Symbol | mk.Expr) -> mk.Symbol | mk.Expr:
    """The str of pl.Expr use brackets for binary expressions."""
    if not isinstance(expr, mk.Expr):
        return expr
    if expr.head is mk.Head.list and len(expr.args) == 1:
        return _unwrap_rust_expr(expr.args[0])
    return mk.Expr(expr.head, [_unwrap_rust_expr(a) for a in expr.args])


def _replace_dyn(string: str, fmt: str) -> str:
    _float_or_int = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    return re.sub(rf"\(dyn {fmt}: ({_float_or_int})\)", r"\1", string)


def _replace_string(string: str) -> str:
    return re.sub(r"String\(([^)]+)\)", r"'\1'", string)


def _polars_expr_to_str(expr: pl.Expr) -> str:
    expr_str = str(expr).replace(".when(", "when(").replace(".strict_cast(", ".cast(")
    if "String(" in expr_str:
        expr_str = _replace_string(expr_str)
    if "dyn int:" in expr_str:
        expr_str = _replace_dyn(expr_str, "int")
    if "dyn float:" in expr_str:
        expr_str = _replace_dyn(expr_str, "float")
    if "[" in expr_str:
        # converting binary expression to str will add brackets so remove them
        # e.g. [col("a") == (3)] -> col("a") == (3)
        out = _unwrap_rust_expr(mk.parse(expr_str))
        expr_str = str(out)
        if (
            out.head is mk.Head.binop
            and expr_str.startswith("(")
            and expr_str.endswith(")")
        ):
            expr_str = expr_str[1:-1]
    try:
        ns = dict(**POLARS_NAMESPACE)
        ExprStr(expr_str, ns).eval()
    except Exception:
        raise ValueError(
            f"Expression {expr_str!r} cannot be safely parsed. Please use "
            "str as the input."
        ) from None
    return expr_str


def _validate_expr(expr: str | pl.Expr) -> str:
    if isinstance(expr, str):
        value = ExprStr(expr, POLARS_NAMESPACE).eval()
        if not isinstance(value, (pl.Expr, str)):
            raise TypeError(f"Invalid type: {type(value)}")
        return expr
    elif isinstance(expr, pl.Expr):
        return _polars_expr_to_str(expr)
    else:
        raise TypeError("Input must be string or polars.Expr type.")


def norm_scalar_expr(val) -> pl.Expr:
    if isinstance(val, pl.Expr):
        return val
    elif isinstance(val, str):
        expr = ExprStr(val, POLARS_NAMESPACE).eval()
        if not isinstance(expr, pl.Expr):
            expr = pl.lit(float(expr))
        return expr
    else:
        return pl.lit(float(val))


PolarsExprStr = Annotated[
    str,
    {
        "widget_type": EvalLineEdit,
        "namespace": POLARS_NAMESPACE,
        "validator": _validate_expr,
    },
]


def norm_expr(expr) -> pl.Expr:
    if isinstance(expr, str):
        val = ExprStr(expr, POLARS_NAMESPACE).eval()
    if isinstance(val, pl.Expr):
        return val
    raise TypeError(f"Invalid type {type(val)} during evaluating {expr!r}")


_Logger = getLogger("cylindra")


class timer:
    def __init__(self, name: str | None = None):
        if name is None:
            try:
                name = inspect.stack()[1].function
            except Exception:  # pragma: no cover
                name = "<unknown>"
        self.name = name
        self.start = default_timer()

    def toc(self):
        dt = default_timer() - self.start
        _Logger.print(f"`{self.name}` ({dt:.1f} sec)")


class CmapDict(TypedDict):
    by: str
    limits: tuple[float, float]
    cmap: Any


def add_molecules(
    viewer: napari.Viewer,
    mol: Molecules,
    name: str,
    source: BaseComponent | None = None,
    metadata: dict[str, Any] = {},
    cmap: CmapDict | None = None,
    **kwargs,
) -> MoleculesLayer:
    """Add Molecules object as a point layer."""
    layer = MoleculesLayer.construct(
        mol,
        name=name,
        source=source,
        metadata=metadata,
        cmap=cmap,
        **kwargs,
    )
    return viewer.add_layer(layer)


def change_viewer_focus(
    viewer: napari.Viewer,
    center: Sequence[float],
    scale: float = 1.0,
) -> None:
    center = np.asarray(center)
    v_scale = np.array([r[2] for r in viewer.dims.range])
    viewer.camera.center = center * scale
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.set_current_step(axis=0, value=center[0] / v_scale[0] * scale)
    return None


def plot_seam_search_result(score: np.ndarray, npf: int):
    import matplotlib.pyplot as plt

    imax = np.argmax(score)
    # plot the score
    plt.figure(figsize=(6, 2.4))
    plt.axvline(imax, color="gray", alpha=0.6)
    plt.axhline(score[imax], color="gray", alpha=0.6)
    plt.plot(score)
    plt.xlabel("PF position")
    plt.ylabel("ΔCorr")
    plt.xticks(np.arange(0, 2 * npf + 1, 4))
    plt.title("Score")
    plt.tight_layout()
    plt.show()


def plot_projections(merge: np.ndarray):
    """Projection of the result of `align_averaged`."""
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5))
    axes: Sequence[plt.Axes]
    # normalize
    if merge.dtype.kind == "f":
        merge = np.clip(merge, 0, 1)
    elif merge.dtype.kind in "ui":
        merge = np.clip(merge, 0, 255)
    else:
        raise RuntimeError("dtype not supported.")
    axes[0].imshow(np.max(merge, axis=0))
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[1].imshow(np.max(merge, axis=1))
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    plt.tight_layout()
    plt.show()
    return None


@dataclass
class FscResult:
    """Result of Fourier Shell Correlation."""

    freq: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    scale: nm

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, scale: nm) -> FscResult:
        freq = df["freq"].to_numpy()
        fsc_all = df.select(pl.col("^FSC.*$")).to_numpy()
        fsc_mean = np.mean(fsc_all, axis=1)
        fsc_std = np.std(fsc_all, axis=1)
        return cls(freq, fsc_mean, fsc_std, scale)

    def get_resolution(self, res: float) -> nm:
        freq0 = None
        for i, fsc1 in enumerate(self.mean):
            if fsc1 < res:
                if i == 0:
                    resolution = 0
                    break
                f0 = self.freq[i - 1]
                f1 = self.freq[i]
                fsc0 = self.mean[i - 1]
                freq0 = (res - fsc1) / (fsc0 - fsc1) * (f0 - f1) + f1
                resolution = self.scale / freq0
                break
        else:
            resolution = np.nan
        return resolution

    def plot(self, criteria: list[float] = [0.143, 0.5]):
        import matplotlib.pyplot as plt

        ind = self.freq <= 0.7
        plt.axhline(0.0, color="gray", alpha=0.5, ls="--")
        plt.axhline(1.0, color="gray", alpha=0.5, ls="--")
        for cr in criteria:
            plt.axhline(cr, color="violet", alpha=0.5, ls="--")
        plt.plot(self.freq[ind], self.mean[ind], color="gold")
        plt.fill_between(
            self.freq[ind],
            y1=self.mean[ind] - self.std[ind],
            y2=self.mean[ind] + self.std[ind],
            color="gold",
            alpha=0.3,
        )
        plt.xlabel("Spatial frequence (1/nm)")
        plt.ylabel("FSC")
        plt.ylim(-0.1, 1.1)
        xticks = np.linspace(0, 0.7, 8)
        per_nm = [r"$\infty$"] + [f"{x:.2f}" for x in self.scale / xticks[1:]]
        plt.xticks(xticks, per_nm)
        plt.tight_layout()
        plt.show()


def coordinates_with_extensions(
    spl: CylSpline, n_extend: dict[int, tuple[int, int]]
) -> NDArray[np.int32]:
    model = spl.cylinder_model()
    coords = list[NDArray[np.int32]]()
    ny, npf = model.shape
    for _idx in range(npf):
        _append, _prepend = n_extend.get(_idx, (0, 0))
        if ny + _append + _prepend <= 0:
            continue  # size is zero
        _nth = np.arange(-_prepend, ny + _append, dtype=np.int32)
        _npf = np.full(_nth.size, _idx, dtype=np.int32)
        coords.append(np.stack([_nth, _npf], axis=1))

    return np.concatenate(coords, axis=0)


def get_code_theme(self: MagicTemplate) -> str:
    """Get the theme for CodeEdit using the napari theme."""
    from napari.utils.theme import get_theme

    if (viewer := self.parent_viewer) or (viewer := napari.current_viewer()):
        theme = get_theme(viewer.theme, as_dict=False).syntax_style
    else:
        bg_color = self.native.palette().color(self.native.backgroundRole())
        if bg_color.lightness() > 128:
            theme = "default"
        else:
            theme = "native"
    return theme


def capitalize(s: str):
    """Just used for button texts."""
    return s.replace("_", " ").capitalize()
