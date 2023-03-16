from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING
from numpy.typing import ArrayLike
import polars as pl

from .spline import Spline
from cylindra.const import nm, SplineAttributes as K, Ori

if TYPE_CHECKING:
    Degenerative = Callable[[ArrayLike], Any]


class CylSpline(Spline):
    """A spline object with cylindrical structure."""    

    _local_cache = (K.localprops,)
    _global_cache = (K.globalprops, K.radius, K.orientation)
    
    def __init__(self, degree: int = 3, *, lims: tuple[float, float] = (0., 1.)):
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
