from typing import Annotated
import numpy as np
import impy as ip

from magicclass import (
    do_not_record,
    magicclass,
    field,
    MagicTemplate,
    bind_key,
    set_design,
    abstractapi,
)
from magicclass.undo import undo_callback
from magicclass.ext.pyqtgraph import QtImageCanvas, mouse_event

from cylindra.utils import roundint, centroid, map_coordinates
from cylindra.const import nm, Mode
from ._child_widget import ChildWidget

_FILP_X = ip.slicer.x[::-1]


@magicclass(layout="horizontal")
class SplineFitter(ChildWidget):
    """
    Manually fit splines with longitudinal projections.

    Attributes
    ----------
    num : int
        Spline number in current tomogram.
    pos : int
        Position along the spline.
    err_max : float
        Maximum allowed error (nm) for spline fitting.
    """

    # Manually fit cylinders with spline curve using longitudinal projections

    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)

    def __init__(self) -> None:
        self._max_interval: nm = 50.0
        self.subtomograms: "ip.ImgArray | None" = None

    def __post_init__(self):
        self.shifts: list[np.ndarray] = None
        self.canvas.min_height = 160
        self.canvas.add_infline(pos=[0, 0], degree=90, color="lime", lw=2)
        self.canvas.add_infline(pos=[0, 0], degree=0, color="lime", lw=2)
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        cos = np.cos(theta)
        sin = np.sin(theta)
        kwargs = dict(color="lime", lw=2, ls="--")
        self.canvas.add_curve(cos, sin, **kwargs, antialias=True)
        self.canvas.add_curve(2 * cos, 2 * sin, **kwargs, antialias=True)

        @self.canvas.mouse_clicked.connect
        def _(e: mouse_event.MouseClickEvent):
            if "left" not in e.buttons():
                return
            x, z = e.pos()
            self._update_cross(x, z)

    @bind_key("Esc")
    def _close(self):
        return self.close()

    @magicclass(record=False)
    class footer(MagicTemplate):
        """Select and fit splines."""

        num = abstractapi()
        pos = abstractapi()
        err_max = abstractapi()
        resample_volumes = abstractapi()
        fit = abstractapi()

    @bind_key("Up")
    def _next_pos(self):
        self.pos.value = min(self.pos.value + 1, self.pos.max)
        self._focus_me()

    @bind_key("Down")
    def _prev_pos(self):
        self.pos.value = max(self.pos.value - 1, self.pos.min)
        self._focus_me()

    @bind_key("Right")
    def _next_num(self):
        self.num.value = min(self.num.value + 1, self.num.max)
        self._focus_me()

    @bind_key("Left")
    def _prev_num(self):
        self.num.value = max(self.num.value - 1, self.num.min)
        self._focus_me()

    num = field(int, label="Spline No.", location=footer).with_options(max=0)
    pos = field(int, label="Position", location=footer).with_options(max=0)
    err_max = field(0.5, label="Max. error", location=footer).with_options(step=0.05)

    def _get_shifts(self, _=None):
        if self.shifts is None:
            return np.zeros((1, 2))
        i = self.num.value
        return np.round(self.shifts[i], 3)

    def _get_binsize(self) -> int:
        parent = self._get_main()
        return roundint(parent._reserved_layers.scale / parent.tomogram.scale)

    def _get_max_interval(self, _=None) -> nm:
        return self._max_interval

    @set_design(text="Resample", location=footer)
    @do_not_record
    def resample_volumes(
        self,
        interval: Annotated[nm, {"label": "Max interval (nm)"}] = 50.0,
    ):
        """
        Resample volumes with given interval.

        Parameters
        ----------
        interval : float
            Maximum interval (nm) between spline anchors that will be used to
            sample subtomogram projections.
        """
        self._max_interval = interval
        parent = self._get_main()
        tomo = parent.tomogram
        self.shifts = [None] * len(tomo.splines)
        self.num.max = len(tomo.splines) - 1
        self.num.min = 0
        self.num.value = 0
        self._cylinder_changed()
        self.canvas.contrast_limits = np.percentile(self.subtomograms, [1, 99.9])
        return None

    @set_design(text="Fit", location=footer)
    def fit(
        self,
        i: Annotated[int, {"bind": num}],
        shifts: Annotated[nm, {"bind": _get_shifts}],
        max_interval: Annotated[nm, {"bind": _get_max_interval}] = 50.0,
        err_max: Annotated[nm, {"bind": err_max}] = 0.5,
    ):
        """Fit current spline."""
        shifts = np.asarray(shifts)
        parent = self._get_main()
        _scale = parent.tomogram.scale
        old_spl = parent.tomogram.splines[i]
        parent.tomogram.splines[i] = new_spl = old_spl.shift(
            positions=old_spl.prep_anchor_positions(max_interval=max_interval),
            shifts=shifts * self._get_binsize() * _scale,
            err_max=err_max,
        )
        self._cylinder_changed()
        parent._update_splines_in_images()

        @undo_callback
        def out():
            parent.tomogram.splines[i] = old_spl.copy()
            if old_spl.has_anchors:
                self._cylinder_changed()
            else:
                del self.canvas.image
            self._get_main()._update_splines_in_images()

        @out.with_redo
        def out():
            parent.tomogram.splines[i] = new_spl
            self._cylinder_changed()
            self._get_main()._update_splines_in_images()

        return out

    def _update_cross(self, x: float, z: float):
        i = self.num.value
        j = self.pos.value
        if self.shifts[i] is None:
            return
        binsize = self._get_binsize()
        itemv = self.canvas.layers[0]
        itemh = self.canvas.layers[1]
        item_circ_inner = self.canvas.layers[2]
        item_circ_outer = self.canvas.layers[3]

        tomo = self._get_main().tomogram
        lz, lx = self.subtomograms.sizesof("zx")
        if not (0 <= x < lx and 0 <= z < lz):
            return
        itemv.pos = [x, z]
        itemh.pos = [x, z]
        spl = tomo.splines[i]
        r_max: nm = spl.config.fit_width / 2
        nbin = max(roundint(r_max / tomo.scale / binsize / 2), 8)
        prof = self.subtomograms[j].radial_profile(
            center=[z, x], nbin=nbin, r_max=r_max
        )
        _scale = self.subtomograms.scale.x
        imax = np.argmax(prof)
        imax_sub = centroid(prof, imax - 5, imax + 5)
        r_peak: nm = (imax_sub + 0.5) / nbin * r_max
        r_inner = max(r_peak - spl.config.thickness_inner, 0) / _scale
        r_outer = (r_peak + spl.config.thickness_outer) / _scale

        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        item_circ_inner.xdata = r_inner * np.cos(theta) + x
        item_circ_inner.ydata = r_inner * np.sin(theta) + z
        item_circ_outer.xdata = r_outer * np.cos(theta) + x
        item_circ_outer.ydata = r_outer * np.sin(theta) + z

        self.shifts[i][j, :] = z - lz / 2 + 0.5, x - lx / 2 + 0.5
        return None

    @num.connect
    def _cylinder_changed(self):
        i: int = self.num.value
        self.pos.value = 0
        parent = self._get_main()
        imgb = parent._reserved_layers.image_data
        tomo = parent.tomogram

        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]
        anc = spl.prep_anchor_positions(max_interval=self._max_interval)
        npos = anc.size
        if self.shifts is None:
            self.shifts = [None] * len(tomo.splines)

        self.shifts[i] = np.zeros((npos, 2))

        binsize = self._get_binsize()
        length_px = tomo.nm2pixel(spl.config.fit_depth, binsize=binsize)
        width_px = tomo.nm2pixel(spl.config.fit_width, binsize=binsize)

        mole = spl.anchors_to_molecules(positions=anc)
        coords = mole.local_coordinates(
            shape=(width_px, length_px, width_px),
            scale=tomo.scale * binsize,
        )
        out = list[ip.ImgArray]()
        for crds in coords:
            out.append(
                map_coordinates(imgb, crds, order=1, mode=Mode.constant, cval=np.mean)
            )
        subtomo = ip.asarray(np.stack(out, axis=0), axes="pzyx").set_scale(
            imgb, unit="nm"
        )
        self.subtomograms = subtomo.mean(axis="y")[_FILP_X]

        self.canvas.image = self.subtomograms[0]
        self.pos.max = npos - 1
        self.canvas.xlim = (0, self.canvas.image.shape[1])
        self.canvas.ylim = (0, self.canvas.image.shape[0])
        lz, lx = self.subtomograms.sizesof("zx")
        self._update_cross(lx / 2 - 0.5, lz / 2 - 0.5)

        return None

    @pos.connect
    def _position_changed(self):
        i = self.num.value
        j = self.pos.value
        self.canvas.image = self.subtomograms[j]
        if self.shifts is not None and self.shifts[i] is not None:
            y, x = self.shifts[i][j]
        else:
            y = x = 0
        lz, lx = self.subtomograms.shape[-2:]
        self._update_cross(x + lx / 2 - 0.5, y + lz / 2 - 0.5)

    def _focus_me(self):
        return self.native.setFocus()
