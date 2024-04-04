from typing import Annotated

import impy as ip
import numpy as np
from magicclass import do_not_record, field, magicclass, set_design, vfield
from magicclass.ext.pyqtgraph import QtMultiImageCanvas

from cylindra.components import CylSpline
from cylindra.const import Mode, nm
from cylindra.utils import map_coordinates
from cylindra.widget_utils import capitalize
from cylindra.widgets.subwidgets._child_widget import ChildWidget


@magicclass
class SplineClipper(ChildWidget):
    canvas = field(QtMultiImageCanvas).with_options(nrows=2, ncols=2)
    clip_length = vfield(nm, label="Clip length (nm)", record=False).with_options(
        max=20.0, step=0.1
    )

    # canvas is as follows:
    # none | xy
    # -----+---
    #  yz  | xz

    @property
    def xy_canvas(self):
        return self.canvas[0, 1]

    @property
    def yz_canvas(self):
        return self.canvas[1, 0]

    @property
    def xz_canvas(self):
        return self.canvas[1, 1]

    def __init__(self):
        self._spline: "CylSpline | None" = None  # the spline object to be shown
        self._original_spline: "CylSpline | None" = None
        self._clip_at_start = True
        self._current_lims: tuple[float, float] = (0.0, 1.0)
        self._subtomograms: "ip.ImgArray | None" = None

    def __post_init__(self):
        self.canvas[1, 0].lock_contrast_limits = True
        self.canvas[0, 1].lock_contrast_limits = True
        self.canvas[1, 1].lock_contrast_limits = True
        self.canvas.enabled = False
        curve_kw = {"color": "lime", "lw": 4, "antialias": True}
        line_kw = {"color": "lime", "lw": 1, "ls": ":"}
        self._xy_line = self.xy_canvas.add_curve([0, 0], [0, 1], **curve_kw)
        self._xy_ref = self.xy_canvas.add_infline(pos=[0, 0], degree=0, **line_kw)
        self._yz_line = self.yz_canvas.add_curve([0, 1], [0, 0], **curve_kw)
        self._yz_ref = self.yz_canvas.add_infline(pos=[0, 0], degree=90, **line_kw)
        self._xz_cross = (
            self.xz_canvas.add_infline((0, 0), 0, **line_kw),
            self.xz_canvas.add_infline((0, 0), 90, **line_kw),
        )

    def _get_spline_id(self, *_) -> int:
        return self._get_main().SplineControl.num

    @property
    def current_clip_length(self) -> tuple[nm, nm]:
        restored = self._spline.restore()
        length = restored.length()
        lim0, lim1 = self._original_spline.lims
        clim0, clim1 = self._current_lims
        return (clim0 - lim0) * length, (lim1 - clim1) * length

    @do_not_record
    @set_design(text=capitalize)
    def load_spline(self, spline: Annotated[int, {"bind": _get_spline_id}]):
        """Load the current spline selected in the main widget."""
        parent = self._get_main()
        spl = parent.tomogram.splines[spline]
        self._spline = self._original_spline = spl
        self._current_lims = spl.lims
        self["clip_length"].max = spl.length()
        self._update_canvas()

    @do_not_record
    @set_design(text=capitalize)
    def the_other_side(self):
        """See the other side of the spline."""
        self._clip_at_start = not self._clip_at_start
        self.clip_length = self.current_clip_length[1 - int(self._clip_at_start)]
        self._update_canvas()

    @do_not_record(recursive=False)
    @set_design(text=capitalize)
    def clip_here(self):
        """Clip the spline at the current position."""
        parent = self._get_main()
        try:
            idx = parent.tomogram.splines.index(self._original_spline)
        except ValueError:
            raise ValueError(
                "The spline shown in the spline clipper widget no longer exists. "
                "Please reload a spline."
            )
        parent.clip_spline(self._get_spline_id(), self.current_clip_length)
        self._spline = self._original_spline = parent.tomogram.splines[idx]
        self._current_lims = self._spline.lims

    @clip_length.connect
    def _clip(self, val: nm):
        length = self._original_spline.length()
        lim0, lim1 = self._original_spline.lims
        if self._clip_at_start:
            self._current_lims = (lim0 + val / length, self._current_lims[1])
        else:
            self._current_lims = (self._current_lims[0], lim1 - val / length)

        self._spline = self._original_spline.clip(*self._current_lims)

    @clip_length.connect
    def _update_canvas(self):
        if self._spline is None:
            return None
        parent = self._get_main()
        tomo = parent.tomogram
        spl = self._spline
        binsize: int = parent._current_binsize
        imgb = parent._reserved_layers.image_data

        length_px = tomo.nm2pixel(spl.config.fit_depth, binsize=binsize)
        width_px = tomo.nm2pixel(spl.config.fit_width, binsize=binsize)

        # sample subtomogram at the edge
        mole = spl.anchors_to_molecules([0.0, 1.0])
        index = 0 if self._clip_at_start else mole.count() - 1
        coords = mole.subset(index).local_coordinates(
            shape=(width_px, length_px, width_px),
            scale=tomo.scale * binsize,
        )

        self._subtomograms = ip.asarray(
            map_coordinates(imgb, coords, order=1, mode=Mode.constant, cval=np.mean),
            axes="zyx",
        )
        self._subtomograms.set_scale(xyz=tomo.scale * binsize)
        shape = self._subtomograms.shape
        self.xy_canvas.image = self._subtomograms.mean(axis="z")
        self.yz_canvas.image = self._subtomograms.mean(axis="x")
        self.xz_canvas.image = self._subtomograms.mean(axis="y")

        # update spline curves in canvases
        zc, yc, xc = np.array(shape) / 2 - 0.5
        self._xy_line.xdata = [xc, xc]
        self._xy_ref.pos = [xc, yc]
        self._yz_line.ydata = [zc, zc]
        self._yz_ref.pos = [yc, zc]
        if self._clip_at_start:
            self._xy_line.ydata = [yc, yc * 2 + 0.5]
            self._yz_line.xdata = [yc, yc * 2 + 0.5]
        else:
            self._xy_line.ydata = [-0.5, yc]
            self._yz_line.xdata = [-0.5, yc]
        self._xz_cross[0].pos = (xc, xc)
        self._xz_cross[1].pos = (zc, zc)
