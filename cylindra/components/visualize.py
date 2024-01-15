from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
import polars as pl
from acryo import Molecules

from cylindra.components.spline import CylSpline
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def flat_view(
    mole: Molecules,
    name: str | pl.Expr | None = None,
    spl: CylSpline | None = None,
    colors: str | Callable[[Any], Any] | Iterable[Any] = "viridis",
    ax: Axes | None = None,
):
    """
    Plot molecule features in 2D figure.

    Parameters
    ----------
    mole : Molecules
        Molecules to plot.
    name : str or pl.Expr
        Feature name or expression to plot. If `colors` is already a sequence
        of colors, this parameter is not needed.
    spl : CylSpline, optional
        The source spline of the molecules.
    colors : callable or sequence, optional
        Color of each molecule. If callable, it must take a feature value and
        return a color. If sequence, it must be a sequence of colors. A
        predefined colormap name is also accepted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    nth = mole.features[Mole.nth].to_numpy()
    pf = mole.features[Mole.pf].to_numpy()
    npf = int(pf.max() + 1)
    if spl is not None:
        spacing = spl.props.get_glob(H.spacing)
        rise = np.deg2rad(spl.props.get_glob(H.rise))
        tan = (
            np.tan(rise)
            / spacing
            * (2 * np.pi * spl.radius / npf)
            * spl.config.rise_sign
        )
    else:
        tan = _infer_start_from_molecules(mole) / npf

    y = nth + tan * pf

    def _get_feature_values():
        df = mole.features.select(name)
        return df[df.columns[0]]

    if callable(colors):
        ref_feature = _get_feature_values()
        face_color = [colors(feat) for feat in ref_feature]
    elif isinstance(colors, str):
        from vispy.color import Colormap, get_colormap

        ref_feature = _get_feature_values()
        cmap: Colormap = get_colormap(colors)
        face_color = cmap.map(ref_feature.to_numpy())
    elif hasattr(colors, "__getitem__"):
        face_color = colors
    elif hasattr(colors, "__iter__"):
        face_color = list(colors)
    else:
        raise TypeError(f"`colors` must be a callable or a sequence, got {colors!r}")

    if ax is None:
        _, ax = plt.subplots()

    for i in range(mole.count()):
        center = (pf[i], y[i])
        circ = Circle(center, 0.5, fc=face_color[i], ec="black", lw=0.1)
        ax.add_patch(circ)

    ax.set_xlim(pf.min() - 0.6, pf.max() + 0.6)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_aspect("equal")
    return ax


def _infer_start_from_molecules(mole: Molecules) -> int:
    """Infer cylinder geometry (ny, npf, nrise) from molecules."""
    columns = mole.features.columns
    if not (Mole.pf in columns and Mole.position in columns):
        raise ValueError(
            f"Molecules must have columns {Mole.pf!r} and {Mole.position!r}."
        )
    npf = mole.features[Mole.pf].max() + 1
    nmole = mole.pos.shape[0]
    ny, res = divmod(nmole, npf)
    if res != 0:
        raise ValueError("Molecules are not correctly labeled.")
    spl_pos = mole.features[Mole.position].to_numpy().reshape(ny, npf)
    dy = np.abs(np.mean(np.diff(spl_pos, axis=0)))
    drise = np.mean(np.diff(spl_pos, axis=1))
    return int(np.round(drise * npf / dy))
