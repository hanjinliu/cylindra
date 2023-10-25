from __future__ import annotations
from typing import Annotated, Sequence, TYPE_CHECKING, Any, TypedDict, Union
from dataclasses import dataclass
from timeit import default_timer
import inspect
import re

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
import polars as pl
import matplotlib.pyplot as plt
import macrokit as mk
from magicclass.logging import getLogger
from magicclass.types import ExprStr
from magicclass.widgets import EvalLineEdit
import napari
from napari.utils.theme import get_theme

from acryo import Molecules, TomogramSimulator
from cylindra import utils
from cylindra.const import nm, PropertyNames as H
from cylindra.types import MoleculesLayer
from cylindra.components._base import BaseComponent
from cylindra._config import get_config

if TYPE_CHECKING:
    from cylindra.components import CylTomogram, CylSpline
    from magicclass import MagicTemplate


# namespace used in predicate
POLARS_NAMESPACE = {
    "pl": pl,
    "col": pl.col,
    "when": pl.when,
    "int": int,
    "float": float,
    "str": str,
    "np": np,
    "__builtins__": {},
}


def _validate_expr_or_scalar(expr: str | pl.Expr | int | float) -> str | int | float:
    if isinstance(expr, str):
        value = ExprStr(expr, POLARS_NAMESPACE).eval()
        if isinstance(value, pl.Expr):
            # NOTE: If a polars Expr is given as a string, it is not needed to check
            # using `_polars_expr_to_str`. Return here.
            return value
        expr = value
    if isinstance(expr, (int, float, np.number)):
        return expr
    elif isinstance(expr, pl.Expr):
        return _polars_expr_to_str(expr)
    else:
        raise TypeError(f"Got invalid type: {type(expr)}")


PolarsExprStrOrScalar = Annotated[
    Union[str, int, float],
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


def _replace_utf8(string: str) -> str:
    ptn = re.compile(r"Utf8\(.?+\)")
    m0 = ptn.search(string)
    if m0 is None:
        return string
    sl_0 = slice(None, m0.start())
    sl_mid = slice(m0.start() + 5, m0.end() - 1)
    sl_1 = slice(m0.end(), None)
    return string[sl_0] + '"' + string[sl_mid] + '"' + string[sl_1]


def _polars_expr_to_str(expr: pl.Expr) -> str:
    expr_str = str(expr).replace(".when(", "when(").replace(".strict_cast(", ".cast(")
    if "Utf8" in expr_str:
        # Utf8(xxx) -> "xxx"
        expr_str = _replace_utf8(expr_str)
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
        ExprStr(expr_str, POLARS_NAMESPACE).eval()
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
        raise TypeError(f"Input must be string or polars.Expr type.")


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
            except Exception:
                name = "<unknown>"
        self.name = name
        self.start = default_timer()

    def toc(self, log: bool = True):
        dt = default_timer() - self.start
        if log:
            _Logger.print_html(f"<code>{self.name}</code> ({dt:.1f} sec)")
        else:
            print(f"{self.name} ({dt:.1f} sec)")


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
    app_cfg = get_config()
    kw = dict(
        size=app_cfg.point_size,
        face_color=app_cfg.molecules_color,
        out_of_slice_display=True,
    )
    kw.update(**kwargs)
    layer = MoleculesLayer(mol, name=name, metadata=metadata.copy(), **kw)
    if source is not None:
        layer.source_component = source
    viewer.add_layer(layer)
    if cmap is not None:
        layer.set_colormap(**cmap)
    layer.view_ndim = app_cfg.molecules_ndim
    layer.editable = False
    return layer


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


def plot_fsc(
    freq: np.ndarray,
    fsc_mean: np.ndarray,
    fsc_std: np.ndarray,
    crit: list[float],
    scale: nm,
):
    ind = freq <= 0.7
    plt.axhline(0.0, color="gray", alpha=0.5, ls="--")
    plt.axhline(1.0, color="gray", alpha=0.5, ls="--")
    for cr in crit:
        plt.axhline(cr, color="violet", alpha=0.5, ls="--")
    plt.plot(freq[ind], fsc_mean[ind], color="gold")
    plt.fill_between(
        freq[ind],
        y1=fsc_mean[ind] - fsc_std[ind],
        y2=fsc_mean[ind] + fsc_std[ind],
        color="gold",
        alpha=0.3,
    )
    plt.xlabel("Spatial frequence (1/nm)")
    plt.ylabel("FSC")
    plt.ylim(-0.1, 1.1)
    xticks = np.linspace(0, 0.7, 8)
    per_nm = [r"$\infty$"] + [f"{x:.2f}" for x in scale / xticks[1:]]
    plt.xticks(xticks, per_nm)
    plt.tight_layout()
    plt.show()


def calc_resolution(
    freq: np.ndarray, fsc: np.ndarray, crit: float = 0.143, scale: nm = 1.0
) -> nm:
    """
    Calculate resolution using arrays of frequency and FSC.
    This function uses linear interpolation to find the solution.
    If the inputs are not accepted, 0 will be returned.
    """
    freq0 = None
    for i, fsc1 in enumerate(fsc):
        if fsc1 < crit:
            if i == 0:
                resolution = 0
                break
            f0 = freq[i - 1]
            f1 = freq[i]
            fsc0 = fsc[i - 1]
            freq0 = (crit - fsc1) / (fsc0 - fsc1) * (f0 - f1) + f1
            resolution = scale / freq0
            break
    else:
        resolution = np.nan
    return resolution


def plot_projections(merge: np.ndarray):
    """Projection of the result of `align_averaged`."""
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
    resolution_0143: nm
    resolution_0500: nm


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


class PaintDevice:
    """
    Device used for painting 3D images.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the target image.
    scale : nm
        Scale of the target image.
    """

    def __init__(self, shape: tuple[int, int, int], scale: nm):
        self._shape = shape
        self._scale = scale

    @property
    def scale(self) -> nm:
        return self._scale

    def paint_molecules(
        self,
        template: NDArray[np.float32],
        molecules: Molecules,
        colors: NDArray[np.float32],
    ):
        simulator = TomogramSimulator(order=1, scale=self._scale)
        simulator.add_molecules(molecules, template)
        img = simulator.simulate(self._shape, colormap=colors)
        return img

    def paint_cylinders(self, tomo: CylTomogram, prop: str):
        lbl = np.zeros(self._shape, dtype=np.uint8)
        all_df = tomo.splines.collect_localprops()
        if all_df is None:
            raise ValueError("No local property found.")
        bin_scale = self._scale  # scale of binned reference image
        binsize = utils.roundint(bin_scale / tomo.scale)

        cylinders = list[list[NDArray[np.bool_]]]()
        matrices = list[list[NDArray[np.float32]]]()
        for i, spl in enumerate(tomo.splines):
            depth = spl.props.window_size.get(prop, spl.config.fit_depth)
            lz, ly, lx = (
                utils.roundint(r / bin_scale * 1.73) * 2 + 1
                for r in [15, depth / 2, 15]
            )
            center = np.array([lz, ly, lx]) / 2 + 0.5
            z, _, x = np.indices((lz, ly, lx))
            # Prepare template hollow image
            _sq = (z - lz / 2 - 0.5) ** 2 + (x - lx / 2 - 0.5) ** 2
            domains = list[NDArray[np.bool_]]()
            dist = [-np.inf] + list(spl.distances()) + [np.inf]
            if len(spl.props.get_loc(H.radius, [])) == spl.anchors.size:
                radii = spl.props.get_loc(H.radius)
            elif spl.props.has_glob(H.radius):
                radii = [spl.props.get_glob(H.radius)] * spl.anchors.size
            else:
                raise RuntimeError(f"Radius not found in spline-{i}.")
            for j, rc in enumerate(radii):
                r0 = max(rc - spl.config.thickness_inner, 0.0) / tomo.scale / binsize
                r1 = (rc + spl.config.thickness_outer) / tomo.scale / binsize
                domain = (r0**2 < _sq) & (_sq < r1**2)
                ry = (
                    min(
                        abs(dist[j + 1] - dist[j]) / 2,
                        abs(dist[j + 2] - dist[j + 1]) / 2,
                        depth / 2,
                    )
                    / bin_scale
                    + 0.5
                )

                ry = max(utils.ceilint(ry), 1)
                domain[:, : ly // 2 - ry] = 0
                domain[:, ly // 2 + ry + 1 :] = 0
                domain = domain.astype(np.float32)
                domains.append(domain)

            cylinders.append(domains)
            matrices.append(spl.affine_matrix(center=center, inverse=True))
            yield

        cylinders = np.concatenate(cylinders, axis=0)
        matrices = np.concatenate(matrices, axis=0)

        out = np.empty_like(cylinders)
        for i, (img, matrix) in enumerate(zip(cylinders, matrices, strict=True)):
            out[i] = ndi.affine_transform(img, matrix, order=1, cval=0, prefilter=False)
        out = out > 0.3

        # paint roughly
        for i, crd in enumerate(tomo.splines.iter_anchor_coords()):
            center = crd / bin_scale
            sl = list[slice]()
            outsl = list[slice]()
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape, strict=True):
                _left = int(c - l / 2 - 0.5)
                _right = _left + l
                _sl, _pad = utils.make_slice_and_pad(_left, _right, size)
                sl.append(_sl)
                outsl.append(
                    slice(
                        _pad[0] if _pad[0] > 0 else None,
                        -_pad[1] if _pad[1] > 0 else None,
                    )
                )

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl[sl][out[i][outsl]] = i + 1
            yield

        return lbl


def get_code_theme(self: MagicTemplate) -> str:
    """Get the theme for CodeEdit using the napari theme."""
    try:
        theme = get_theme(self.parent_viewer.theme, as_dict=True)["syntax_style"]
    except Exception:
        bg_color = self.native.palette().color(self.native.backgroundRole())
        if bg_color.lightness() > 128:
            theme = "default"
        else:
            theme = "native"
    return theme


def capitalize(s: str):
    """Just used for button texts."""
    return s.replace("_", " ").capitalize()
