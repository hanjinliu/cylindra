from __future__ import annotations
from types import TracebackType

import weakref
from typing import TYPE_CHECKING, ContextManager, Iterable

from magicclass.undo import undo_callback

if TYPE_CHECKING:
    from cylindra.components import CylSpline, CylTomogram
    from cylindra.widgets.main import CylindraMainWidget


class SplineTracker(ContextManager):
    """Object that tracks changes in splines and allows to undo them."""

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
        self._spline_new = self._get_spline_dict()

    def _get_spline_dict(self) -> dict[int, CylSpline]:
        main = self._widget()
        if main is None:
            raise RuntimeError("Tomogram has been deleted")
        return {i: main.tomogram.splines[i].copy() for i in self._indices}

    def as_undo_callback(self):
        main = self._widget()
        if main is None:
            raise RuntimeError("Tomogram has been deleted")
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


def normalize_spline_indices(indices: Iterable[int], tomo: CylTomogram) -> list[int]:
    indices = list(indices)
    if len(indices) == 0:
        indices = list(range(len(tomo.splines)))
    if len(indices) == 0:
        raise ValueError("No splines to operate on.")
    return indices
