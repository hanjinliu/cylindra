from __future__ import annotations
from types import TracebackType

import weakref
from typing import TYPE_CHECKING, ContextManager, Iterable, Literal
import numpy as np

from magicclass.undo import undo_callback
from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from cylindra.components import CylSpline, CylTomogram
    from cylindra.widgets.main import CylindraMainWidget


class SplineTracker(ContextManager):
    """
    Object that tracks changes in splines and allows to undo them.

    Parameters
    ----------
    widget : CylindraMainWidget
        The main widget. A weak reference of it will be stored.
    indices : iterable of int
        Indices of splines to track.
    sample : bool, optional
        If True, subtomograms will be sampled after undo/redo operations.
    """

    def __init__(
        self, widget: CylindraMainWidget, indices: Iterable[int], sample: bool = False
    ):
        self._widget = weakref.ref(widget)
        self._indices = list(indices)
        self._spline_old: dict[int, CylSpline] | None = None
        self._spline_new: dict[int, CylSpline] | None = None
        self._sample = sample

    def __enter__(self):
        self._spline_old = self._get_spline_dict()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        value: BaseException | None,
        tb: TracebackType | None,
        /,
    ) -> bool | None:
        if exc_type is None:
            self._spline_new = self._get_spline_dict()
        else:
            assert value is not None
            tomo = self._get_main().tomogram
            for i, spl in self._spline_old.items():
                tomo.splines[i] = spl
            raise value

    def _get_main(self) -> CylindraMainWidget:
        main = self._widget()
        if main is None:
            raise RuntimeError("Tomogram has been deleted")
        return main

    def _get_spline_dict(self) -> dict[int, CylSpline]:
        main = self._get_main()
        return {i: main.tomogram.splines[i].copy() for i in self._indices}

    def as_undo_callback(self):
        main = self._get_main()
        tomo = main.tomogram

        @undo_callback
        def undo_op():
            for i, spl in self._spline_old.items():
                tomo.splines[i] = spl.copy()
            main._update_splines_in_images()
            if self._sample:
                main.sample_subtomograms()
            else:
                main._update_local_properties_in_widget()
                main._update_global_properties_in_widget()

        @undo_op.with_redo
        def undo_op():
            for i, spl in self._spline_new.items():
                tomo.splines[i] = spl.copy()
            main._update_splines_in_images()
            if self._sample:
                main.sample_subtomograms()
            else:
                main._update_local_properties_in_widget()
                main._update_global_properties_in_widget()

        return undo_op


def normalize_offsets(
    offsets: tuple[float, float] | None, spl: CylSpline
) -> tuple[float, float]:
    if offsets is None:
        offsets = (
            spl.props.get_glob(H.offset_axial, 0.0),
            spl.props.get_glob(H.offset_angular, 0.0),
        )
    elif len(offsets) != 2:
        raise ValueError(f"offsets must be a tuple of length 2, got {offsets!r}")
    return offsets


def normalize_radius(radius: float | None, spl: CylSpline) -> float:
    if radius is None:
        _r = spl.props.get_glob(H.offset_radial, 0.0) + spl.radius
    else:
        _r = radius
    return _r


def rotvec_from_axis_and_degree(axis: Literal["z", "y", "x"], deg: float):
    if axis == "z":
        unit_vec = np.array([1, 0, 0], dtype=np.float32)
    elif axis == "y":
        unit_vec = np.array([0, 1, 0], dtype=np.float32)
    elif axis == "x":
        unit_vec = np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown axis: {axis!r}")
    return unit_vec * np.deg2rad(deg)
