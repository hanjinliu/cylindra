from __future__ import annotations

from typing import Any
import numpy as np
import polars as pl

from ._spline_base import Spline
from cylindra.const import (
    nm,
    Ori,
    PropertyNames as H,
)
from cylindra.utils import roundint
from cylindra.components.cylindric import CylinderModel
from cylindra.components._boundary import solve_cylinder, CylindricParameters


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
        if value <= 0:
            raise ValueError("Radius must be positive.")
        self.props.update_glob([pl.Series(H.radius, [value], dtype=pl.Float32)])
        return None

    def radius_range(self, rc: nm | None = None) -> tuple[nm, nm]:
        """Return the range of the radius used for the cylindric coordinate."""
        if rc is None:
            if self.radius is None:
                raise ValueError("Radius is not set.")
            rc = self.radius
        cfg = self.config
        return (max(rc - cfg.thickness_inner, 0.0), rc + cfg.thickness_outer)

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
        return self.cylinder_params(**kwargs).start

    def cylinder_params(self, **kwargs) -> CylindricParameters:
        """Get the cylinder parameters of the spline."""
        radius = _get_globalprops(self, kwargs, H.radius)
        if radius is None:
            raise ValueError("Radius is not set.")
        radius += (self.config.thickness_outer - self.config.thickness_inner) / 2
        return solve_cylinder(
            spacing=_get_globalprops(self, kwargs, H.spacing),
            dimer_twist=_get_globalprops(self, kwargs, H.dimer_twist),
            skew=_get_globalprops(self, kwargs, H.skew),
            rise_angle=_get_globalprops(self, kwargs, H.rise),
            radius=radius,
            npf=_get_globalprops(self, kwargs, H.npf),
            start=_get_globalprops(self, kwargs, H.start),
            allow_duplicate=True,
            rise_sign=self.config.rise_sign,
        )

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
        cp = self.cylinder_params(**kwargs)
        ly = cp.spacing_proj
        la = cp.lat_spacing_proj
        factor = ly / la
        ny = roundint(length / ly) + 1  # number of monomers in y-direction

        if offsets is None:
            offsets = (0.0, 0.0)

        return CylinderModel(
            shape=(ny, cp.npf),
            tilts=(
                cp.tan_skew * factor,
                cp.tan_rise / factor * self.config.rise_sign,
            ),
            intervals=(ly, la / cp.perimeter * 2 * np.pi),
            radius=cp.radius,
            offsets=offsets,
        )

    def update_props(
        self,
        *,
        npf: int | None = None,
        orientation: Ori | str | None = None,
    ):
        """Update the npf or orientation parameters in place."""
        loc = list[pl.Expr]()
        glob = list[pl.Series]()
        if npf is not None:
            loc.append(pl.repeat(npf, pl.count()).cast(pl.UInt8).alias(H.npf))
            glob.append(pl.Series([npf]).cast(pl.UInt8).alias(H.npf))
        if orientation is not None:
            glob.append(
                pl.Series([str(orientation)]).cast(pl.Utf8).alias(H.orientation)
            )

        ldf = self.localprops.with_columns(loc)
        gdf = self.globalprops.with_columns(glob)

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


def _get_globalprops(spl: CylSpline, kwargs: dict[str, Any], name: str):
    if name in kwargs:
        return kwargs[name]
    return spl.props.get_glob(name, None)
