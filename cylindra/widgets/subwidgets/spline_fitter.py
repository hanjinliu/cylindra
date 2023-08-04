from typing import Annotated
import numpy as np
import impy as ip

from magicclass import (
    magicclass,
    field,
    MagicTemplate,
    bind_key,
    set_design,
    abstractapi,
)
from magicclass.types import Bound
from magicclass.undo import undo_callback
from magicclass.ext.pyqtgraph import QtImageCanvas, mouse_event

from cylindra.utils import roundint, centroid, map_coordinates
from cylindra.const import nm, GlobalVariables as GVar, Mode


@magicclass
class SplineFitter(MagicTemplate):
    # Manually fit cylinders with spline curve using longitudinal projections

    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)

    def __init__(self) -> None:
        self._max_interval: nm = None
        self.subtomograms: ip.ImgArray = None

    @magicclass(layout="horizontal", record=False)
    class controller(MagicTemplate):
        """
        Control spline positions.

        Attributes
        ----------
        num : int
            Spline number in current tomogram.
        pos : int
            Position along the spline.
        """

        num = field(int, label="Spline No.").with_options(max=0)
        pos = field(int, label="Position").with_options(max=0)
        fit = abstractapi()

        @bind_key("Up")
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)

        @bind_key("Down")
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)

    def _get_shifts(self, _=None):
        if self.shifts is None:
            return np.zeros(3)
        i = self.controller.num.value
        return np.round(self.shifts[i], 3)

    def _get_binsize(self) -> int:
        parent = self._get_parent()
        return roundint(parent._reserved_layers.scale / parent.tomogram.scale)

    def _get_max_interval(self, _=None):
        return self._max_interval

    @controller.wraps
    @set_design(text="Fit")
    def fit(
        self,
        shifts: Bound[_get_shifts],
        i: Annotated[int, {"bind": controller.num}],
        max_interval: Bound[_get_max_interval] = 50.0,
    ):
        """Fit current spline."""
        shifts = np.asarray(shifts)
        parent = self._get_parent()
        _scale = parent.tomogram.scale
        old_spl = parent.tomogram.splines[i]

        min_cr = GVar.min_curvature_radius
        new_spl = (
            old_spl.make_anchors(max_interval=max_interval)
            .shift_coa(
                shifts=shifts * self._get_binsize() * _scale,
                min_radius=min_cr,
                weight_ramp=(min_cr / 10, 0.5),
            )
            .make_anchors(max_interval=max_interval)
        )
        parent.tomogram.splines[i] = new_spl
        self._cylinder_changed()
        parent._update_splines_in_images()

        @undo_callback
        def out():
            parent.tomogram.splines[i] = old_spl.copy()
            if old_spl.has_anchors:
                self._cylinder_changed()
            else:
                del self.canvas.image
            self._get_parent()._update_splines_in_images()

        @out.with_redo
        def out():
            parent.tomogram.splines[i] = new_spl
            self._cylinder_changed()
            self._get_parent()._update_splines_in_images()

        return out

    def __post_init__(self):
        self.shifts: list[np.ndarray] = None
        self.canvas.min_height = 160
        self.canvas.add_infline(pos=[0, 0], degree=90, color="lime", lw=2)
        self.canvas.add_infline(pos=[0, 0], degree=0, color="lime", lw=2)
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        cos = np.cos(theta)
        sin = np.sin(theta)
        self.canvas.add_curve(cos, sin, color="lime", lw=2, ls="--")
        self.canvas.add_curve(2 * cos, 2 * sin, color="lime", lw=2, ls="--")
        self.controller.max_height = 50
        self.controller.height = 50

        @self.canvas.mouse_clicked.connect
        def _(e: mouse_event.MouseClickEvent):
            if "left" not in e.buttons():
                return
            x, z = e.pos()
            self._update_cross(x, z)

    def _get_parent(self):
        from cylindra.widgets.main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    def _update_cross(self, x: float, z: float):
        i = self.controller.num.value
        j = self.controller.pos.value
        if self.shifts[i] is None:
            return
        binsize = self._get_binsize()
        itemv = self.canvas.layers[0]
        itemh = self.canvas.layers[1]
        item_circ_inner = self.canvas.layers[2]
        item_circ_outer = self.canvas.layers[3]
        itemv.pos = [x, z]
        itemh.pos = [x, z]

        tomo = self._get_parent().tomogram
        r_max: nm = GVar.fit_width / 2
        nbin = max(roundint(r_max / tomo.scale / binsize / 2), 8)
        prof = self.subtomograms[j].radial_profile(
            center=[z, x], nbin=nbin, r_max=r_max
        )
        imax = np.argmax(prof)
        imax_sub = centroid(prof, imax - 5, imax + 5)
        r_peak = (imax_sub + 0.5) / nbin * r_max
        r_inner = max(r_peak - GVar.thickness_inner, 0) / tomo.scale / binsize
        r_outer = (r_peak + GVar.thickness_outer) / tomo.scale / binsize

        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        item_circ_inner.xdata = r_inner * np.cos(theta) + x
        item_circ_inner.ydata = r_inner * np.sin(theta) + z
        item_circ_outer.xdata = r_outer * np.cos(theta) + x
        item_circ_outer.ydata = r_outer * np.sin(theta) + z

        lz, lx = self.subtomograms.sizesof("zx")
        self.shifts[i][j, :] = z - lz / 2 + 0.5, x - lx / 2 + 0.5
        return None

    def _load_parent_state(self, max_interval: nm):
        self._max_interval = max_interval
        parent = self._get_parent()
        tomo = parent.tomogram
        for spl in tomo.splines:
            spl.make_anchors(max_interval=self._max_interval)

        self.shifts = [None] * len(tomo.splines)
        self.controller.num.max = len(tomo.splines) - 1
        self.controller.num.min = 0
        self.controller.num.value = 0
        self._cylinder_changed()

    @controller.num.connect
    def _cylinder_changed(self):
        i: int = self.controller.num.value
        self.controller.pos.value = 0
        parent = self._get_parent()
        imgb = parent._reserved_layers.image_data
        tomo = parent.tomogram

        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]
        if not spl.has_anchors:
            return
        npos = spl.anchors.size
        if self.shifts is None:
            self.shifts = [None] * len(tomo.splines)

        self.shifts[i] = np.zeros((npos, 2))

        binsize = self._get_binsize()
        length_px = tomo.nm2pixel(GVar.fit_depth, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fit_width, binsize=binsize)

        mole = spl.anchors_to_molecules()
        coords = mole.local_coordinates(
            shape=(width_px, length_px, width_px),
            scale=tomo.scale * binsize,
        )
        out = list[ip.ImgArray]()
        for crds in coords:
            out.append(
                map_coordinates(imgb, crds, order=1, mode=Mode.constant, cval=np.mean)
            )
        subtomo = ip.asarray(np.stack(out, axis=0), axes="pzyx")
        self.subtomograms = subtomo.proj("y")[ip.slicer.x[::-1]]

        self.canvas.image = self.subtomograms[0]
        self.controller.pos.max = npos - 1
        self.canvas.xlim = (0, self.canvas.image.shape[1])
        self.canvas.ylim = (0, self.canvas.image.shape[0])
        lz, lx = self.subtomograms.sizesof("zx")
        self._update_cross(lx / 2 - 0.5, lz / 2 - 0.5)

        return None

    @controller.pos.connect
    def _position_changed(self):
        i = self.controller.num.value
        j = self.controller.pos.value
        self.canvas.image = self.subtomograms[j]
        if self.shifts is not None and self.shifts[i] is not None:
            y, x = self.shifts[i][j]
        else:
            y = x = 0
        lz, lx = self.subtomograms.shape[-2:]
        self._update_cross(x + lx / 2 - 0.5, y + lz / 2 - 0.5)
