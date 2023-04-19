from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
import polars as pl

from .spline import Spline
from cylindra.const import nm, SplineAttributes as K, Ori, PropertyNames as H
from cylindra.utils import roundint
from cylindra.components.cylindric import CylinderModel

if TYPE_CHECKING:
    Degenerative = Callable[[ArrayLike], Any]


class CylSpline(Spline):
    """A spline object with cylindrical structure."""

    _local_cache = (K.localprops,)
    _global_cache = (K.globalprops, K.radius, K.orientation)

    def __init__(self, degree: int = 3, *, lims: tuple[float, float] = (0.0, 1.0)):
        """
        Spline object for a cylinder.

        Parameters
        ----------
        k : int, default is 3
            Spline order.
        """
        super().__init__(degree=degree, lims=lims)
        self.orientation = Ori.none
        self.radius: nm | None = None
        self.localprops: pl.DataFrame | None = None
        self.globalprops: pl.DataFrame | None = None

    def invert(self) -> CylSpline:
        """
        Invert the direction of spline. Also invert orientation if exists.

        Returns
        -------
        CylSpline
            Inverted object
        """
        inverted = super().invert()
        inverted.radius = self.radius
        if self.localprops is not None:
            inverted.localprops = self.localprops[::-1]
        inverted.globalprops = self.globalprops
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

        clipped.radius = self.radius
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

    @property
    def orientation(self) -> Ori:
        """Orientation of spline."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: Ori | str | None):
        """Set orientation of spline."""
        if value is None:
            self._orientation = Ori.none
        else:
            self._orientation = Ori(value)

    def cylinder_model(
        self,
        offsets: tuple[float, float] = (
            0.0,
            0.0,
        ),
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

        if all(k in kwargs for k in [H.yPitch, H.skewAngle, H.riseAngle, H.nPF]):
            props = kwargs
        else:
            # Get structural parameters
            props = self.globalprops
            if props is None:
                raise ValueError("No global properties are set.")
            props = {
                k: props[k][0] for k in [H.yPitch, H.skewAngle, H.riseAngle, H.nPF]
            }

        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        radius = kwargs.get("radius", self.radius)

        ny = roundint(length / pitch)  # number of monomers in y-direction
        tan_rise = np.tan(np.deg2rad(rise))

        # Construct meshgrid
        # a-coordinate must be radian.
        # If starting number is non-integer, we must determine the seam position to correctly
        # map monomers. Use integer here.
        tilts = (
            np.deg2rad(skew) / (4 * np.pi) * npf,
            roundint(-tan_rise * 2 * np.pi * radius / pitch) / npf,
        )

        if offsets is None:
            offsets = (0.0, 0.0)

        return CylinderModel(
            shape=(ny, npf),
            tilts=tilts,
            interval=pitch,
            radius=radius,
            offsets=offsets,
        )
