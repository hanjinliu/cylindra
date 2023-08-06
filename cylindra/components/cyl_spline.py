from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl

from .spline import Spline
from cylindra.const import (
    nm,
    Ori,
    PropertyNames as H,
    GlobalVariables as GVar,
)
from cylindra.utils import roundint
from cylindra.components.cylindric import CylinderModel
from cylindra.components._boundary import solve_cylinder


class CylSpline(Spline):
    """A spline object with cylindrical structure."""

    @property
    def radius(self) -> nm | None:
        """Average radius of the cylinder."""
        return self.props.get_glob(H.radius, None)

    @radius.setter
    def radius(self, value: nm | None):
        if value is None:
            if H.radius in self.globalprops.columns:
                self.props.drop_glob(H.radius)
            return None
        self.props.update_glob([pl.Series(H.radius, [value], dtype=pl.Float32)])
        return None

    @property
    def orientation(self) -> Ori:
        """Orientation of the spline."""
        return Ori(str(self.props.get_glob(H.orientation, "none")))

    @orientation.setter
    def orientation(self, value: Ori | str | None):
        if value is None:
            value = Ori.none
        else:
            value = Ori(value)
        self.props.update_glob([pl.Series(H.orientation, [str(value)])])
        return None

    def invert(self) -> CylSpline:
        """
        Invert the direction of spline. Also invert orientation if exists.

        Returns
        -------
        CylSpline
            Inverted object
        """
        # NOTE: invert() calls clip() internally.
        # We don't have to invert the orientation here.
        new = super().invert()
        new.props.update_loc(self.props.loc[::-1], self.props.window_size)
        return new

    def clip(self, start: float, stop: float) -> CylSpline:
        """
        Clip spline and generate a new one.

        This method does not convert spline bases. ``_lims`` is updated instead.
        For instance, if you want to clip spline at 20% to 80% position, call
        ``spl.clip(0.2, 0.8)``. If ``stop < start``, the orientation of spline
        will be inverted, thus the ``orientation`` attribute will also be inverted.

        Parameters
        ----------
        start : float
            New starting position.
        stop : float
            New stopping position.

        Returns
        -------
        CylSpline
            Clipped spline.
        """
        clipped = super().clip(start, stop)

        clipped.props.glob = self.props.glob.clone()
        if start > stop:
            clipped.orientation = Ori.invert(self.orientation)
        else:
            clipped.orientation = self.orientation
        return clipped

    def restore(self) -> CylSpline:
        """
        Restore the original, not-clipped spline.

        Returns
        -------
        Spline
            Copy of the original spline.
        """
        original = super().restore()
        start, stop = self._lims
        if start > stop:
            original.orientation = Ori.invert(self.orientation)
        else:
            original.orientation = self.orientation
        return original

    def nrise(self, **kwargs):
        return solve_cylinder(
            spacing=_get_globalprops(self, kwargs, H.spacing),
            skew_angle=_get_globalprops(self, kwargs, H.skew),
            skew_tilt_angle=_get_globalprops(self, kwargs, H.skew),
            rise_angle=_get_globalprops(self, kwargs, H.rise) * GVar.rise_sign,
            radius=_get_globalprops(self, kwargs, H.radius),
            npf=roundint(_get_globalprops(self, kwargs, H.npf)),
            start=_get_globalprops(self, kwargs, H.start),
            allow_duplicate=True,
            rise_sign=GVar.rise_sign,
        ).start

    def cylinder_model(
        self,
        offsets: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> CylinderModel:
        """
        Return the cylinder model of the spline.

        Parameters
        ----------
        offsets : tuple of float, optional
            Offset of the model. See :meth:`map_monomers` for details.

        Returns
        -------
        CylinderModel
            The cylinder model.
        """
        length = self.length()
        cp = solve_cylinder(
            spacing=_get_globalprops(self, kwargs, H.spacing),
            skew_angle=_get_globalprops(self, kwargs, H.skew),
            skew_tilt_angle=_get_globalprops(self, kwargs, H.skew),
            rise_angle=_get_globalprops(self, kwargs, H.rise),
            radius=_get_globalprops(self, kwargs, H.radius),
            npf=_get_globalprops(self, kwargs, H.npf),
            start=_get_globalprops(self, kwargs, H.start),
            allow_duplicate=True,
            rise_sign=GVar.rise_sign,
        )

        factor = cp.spacing / (cp.perimeter / cp.npf)
        ny = roundint(length / cp.spacing) + 1  # number of monomers in y-direction

        if offsets is None:
            offsets = (0.0, 0.0)

        skew_incr = cp.radius * cp.skew_angle_rad * cp.start / 2 * GVar.rise_sign
        return CylinderModel(
            shape=(ny, cp.npf),
            tilts=(cp.tan_skew_tilt * factor, cp.tan_rise / factor * GVar.rise_sign),
            intervals=(cp.spacing, (cp.perimeter + skew_incr) / cp.radius / cp.npf),
            radius=cp.radius,
            offsets=offsets,
        )

    def update_props(
        self,
        *,
        spacing: nm | None = None,
        skew: float | None = None,
        rise: float | None = None,
        npf: int | None = None,
        radius: nm | None = None,
        orientation: Ori | str | None = None,
    ):
        loc = list[pl.Expr]()
        glob = list[pl.Series]()
        if spacing is not None:
            loc.append(pl.repeat(spacing, pl.count()).cast(pl.Float32).alias(H.spacing))
            glob.append(pl.Series([spacing]).cast(pl.Float32).alias(H.spacing))
        if skew is not None:
            loc.append(pl.repeat(skew, pl.count()).cast(pl.Float32).alias(H.skew))
            glob.append(pl.Series([skew]).cast(pl.Float32).alias(H.skew))
        if rise is not None:
            loc.append(pl.repeat(rise, pl.count()).cast(pl.Float32).alias(H.rise))
            glob.append(pl.Series([rise]).cast(pl.Float32).alias(H.rise))
        if npf is not None:
            loc.append(pl.repeat(npf, pl.count()).cast(pl.UInt8).alias(H.npf))
            glob.append(pl.Series([npf]).cast(pl.UInt8).alias(H.npf))
        if radius is not None:
            glob.append(pl.Series([radius]).cast(pl.Float32).alias(H.radius))
        if orientation is not None:
            glob.append(
                pl.Series([str(orientation)]).cast(pl.Utf8).alias(H.orientation)
            )

        ldf = self.localprops.with_columns(loc)
        gdf = self.globalprops.with_columns(glob)

        # update H.start
        if rise is not None:
            r = radius if radius is not None else self.radius
            if r is not None and self.props.has_loc([H.rise, H.spacing, H.skew]):
                _start_loc = rise_to_start(
                    rise=np.deg2rad(ldf[H.rise].to_numpy()),
                    space=ldf[H.spacing].to_numpy(),
                    skew=np.deg2rad(ldf[H.skew].to_numpy()),
                    perimeter=2 * r * np.pi,
                )
                ldf = ldf.with_columns(
                    pl.Series(_start_loc).cast(pl.Float32).alias(H.start)
                )
            if r is not None and self.props.has_glob([H.rise, H.spacing, H.skew]):
                _start_glob = rise_to_start(
                    rise=np.deg2rad(gdf[H.rise].to_numpy()),
                    space=gdf[H.spacing].to_numpy(),
                    skew=np.deg2rad(gdf[H.skew].to_numpy()),
                    perimeter=2 * r * np.pi,
                )
                gdf = gdf.with_columns(
                    pl.Series(_start_glob).cast(pl.Float32).alias(H.start)
                )

        self.props.loc = ldf
        self.props.glob = gdf
        return self

    def _need_rotation(self, orientation: Ori | str | None) -> bool:
        if orientation is not None:
            orientation = Ori(orientation)
            if orientation is Ori.none or self.orientation is Ori.none:
                raise ValueError(
                    "Either molecules' orientation or the input orientation should "
                    "not be none."
                )
            if orientation is not self.orientation:
                return True
        return False


def rise_to_start(rise: float, space: nm, skew: float, perimeter: nm) -> float:
    """Convert rise angle to start number."""
    tan_rise = np.tan(rise)
    return perimeter / space / (np.tan(skew) * tan_rise + 1) * tan_rise


def _get_globalprops(spl: CylSpline, kwargs: dict[str, Any], name: str):
    if name in kwargs:
        return kwargs[name]
    return spl.props.get_glob(name, None)
