from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
import polars as pl

from .spline import Spline
from cylindra.const import (
    nm,
    Ori,
    PropertyNames as H,
)
from cylindra.utils import roundint
from cylindra.components.cylindric import CylinderModel

if TYPE_CHECKING:
    Degenerative = Callable[[ArrayLike], Any]


class CylSpline(Spline):
    """A spline object with cylindrical structure."""

    @property
    def radius(self) -> nm | None:
        """Average radius of the cylinder."""
        return self.get_globalprops(H.radius, None)

    @radius.setter
    def radius(self, value: nm | None):
        if value is None:
            if H.radius in self.globalprops.columns:
                self.globalprops = self.globalprops.drop(H.radius)
            return None
        col = pl.Series(H.radius, [value]).cast(pl.Float32)
        self.globalprops = self.globalprops.with_columns(col)
        return None

    @property
    def orientation(self) -> Ori:
        """Orientation of the spline."""
        return Ori(str(self.get_globalprops(H.orientation, "none")))

    @orientation.setter
    def orientation(self, value: Ori | str | None):
        if value is None:
            value = Ori.none
        else:
            value = Ori(value)
        col = pl.Series(H.orientation, [str(value)])
        self.globalprops = self.globalprops.with_columns(col)
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
        inverted = super().invert()
        inverted.localprops = self.localprops[::-1]

        return inverted

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

        clipped.globalprops = self.globalprops
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
        interv = (
            self.get_globalprops(H.yPitch)
            if H.yPitch not in kwargs
            else kwargs[H.yPitch]
        )
        skew = (
            self.get_globalprops(H.skewAngle)
            if H.skewAngle not in kwargs
            else kwargs[H.skewAngle]
        )
        rise = (
            -self.get_globalprops(H.riseAngle)
            if H.riseAngle not in kwargs
            else kwargs[H.riseAngle]
        )
        npf = (
            roundint(self.get_globalprops(H.nPF))
            if H.nPF not in kwargs
            else kwargs[H.nPF]
        )
        radius = (
            self.get_globalprops(H.radius)
            if H.radius not in kwargs
            else kwargs[H.radius]
        )

        perimeter = 2 * np.pi * radius
        rise_rad = np.deg2rad(rise)
        skew_rad = np.deg2rad(skew)

        nrise = roundint(
            perimeter
            * np.tan(rise_rad)
            / (interv - radius * skew_rad * np.tan(rise_rad) / 2)
        )
        if nrise == 0:
            tan_rise = 0
            tan_skew = radius * skew_rad / interv / 2
            skew_incr = 0
        else:
            pitch_incr = nrise * interv
            skew_incr = radius * skew_rad * nrise / 2

            tan_rise = pitch_incr / (perimeter + skew_incr)
            tan_skew = skew_incr / pitch_incr

        factor = interv / (perimeter / npf)

        ny = roundint(length / interv) + 1  # number of monomers in y-direction

        if offsets is None:
            offsets = (0.0, 0.0)

        return CylinderModel(
            shape=(ny, npf),
            tilts=(tan_skew * factor, tan_rise / factor),
            intervals=(interv, (perimeter + skew_incr) / perimeter * np.pi * 2 / npf),
            radius=radius,
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
        loc = []
        glob = []
        if spacing is not None:
            loc.append(pl.repeat(spacing, pl.count()).cast(pl.Float32).alias(H.yPitch))
            glob.append(pl.Series([spacing]).cast(pl.Float32).alias(H.yPitch))
        if skew is not None:
            loc.append(pl.repeat(skew, pl.count()).cast(pl.Float32).alias(H.skewAngle))
            glob.append(pl.Series([skew]).cast(pl.Float32).alias(H.skewAngle))
        if rise is not None:
            loc.append(pl.repeat(rise, pl.count()).cast(pl.Float32).alias(H.riseAngle))
            glob.append(pl.Series([rise]).cast(pl.Float32).alias(H.riseAngle))
        if npf is not None:
            loc.append(pl.repeat(npf, pl.count()).cast(pl.UInt8).alias(H.nPF))
            glob.append(pl.Series([npf]).cast(pl.UInt8).alias(H.nPF))
        if radius is not None:
            glob.append(pl.Series([radius]).cast(pl.Float32).alias(H.radius))
        if orientation is not None:
            glob.append(pl.Series([orientation]).cast(pl.Utf8).alias(H.orientation))

        self.localprops = self.localprops.with_columns(loc)
        self.globalprops = self.globalprops.with_columns(glob)
        return self
