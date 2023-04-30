from magicclass import magicclass, MagicTemplate, field, vfield, do_not_record
from magicclass.types import Bound
from magicclass.ext.pyqtgraph import QtMultiImageCanvas
from cylindra.const import GlobalVariables as GVar, Mode, nm
from cylindra.utils import map_coordinates
import numpy as np
import impy as ip


@magicclass
class SplineClipper(MagicTemplate):
    canvas = field(QtMultiImageCanvas).with_options(nrows=2, ncols=2)
    clip_length = vfield(nm, label="Clip length (nm)", record=False).with_options(
        max=20, step=0.1
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
        self._spline = None
        self._clip_at_start = True
        self._original_lims: tuple[float, float] = (0.0, 1.0)
        self._current_lims: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        self.canvas[1, 0].lock_contrast_limits = True
        self.canvas[0, 1].lock_contrast_limits = True
        self.canvas[1, 1].lock_contrast_limits = True
        self.canvas.enabled = False

        self._xy_line = self.xy_canvas.add_curve([0, 0], [0, 1], color="lime", lw=4)
        self._xy_ref = self.xy_canvas.add_infline(
            pos=[0, 0], degree=0, color="lime", lw=1, ls=":"
        )
        self._yz_line = self.yz_canvas.add_curve([0, 1], [0, 0], color="lime", lw=4)
        self._yz_ref = self.yz_canvas.add_infline(
            pos=[0, 0], degree=90, color="lime", lw=1, ls=":"
        )
        self._xz_cross = (
            self.xz_canvas.add_infline((0, 0), 0, color="lime", lw=1, ls=":"),
            self.xz_canvas.add_infline((0, 0), 90, color="lime", lw=1, ls=":"),
        )

    def _parent_widget(self):
        from cylindra.widgets import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    def _get_spline_id(self, *_) -> int:
        return self._parent_widget().SplineControl.num

    @property
    def current_clip_length(self) -> tuple[nm, nm]:
        restored = self._spline.restore()
        length = restored.length()
        lim0, lim1 = self._original_lims
        clim0, clim1 = self._current_lims
        return (clim0 - lim0) * length, (lim1 - clim1) * length

    @do_not_record
    def load_spline(self, spline: Bound[_get_spline_id]):
        parent = self._parent_widget()
        spl = parent.tomogram.splines[spline]
        self._spline = spl
        self._original_lims = self._current_lims = spl.lims
        self._subtomogram: "ip.ImgArray | None" = None
        self["clip_length"].max = spl.length()
        self._update_canvas()

    @do_not_record
    def the_other_side(self):
        self._clip_at_start = not self._clip_at_start
        self.clip_length = self.current_clip_length[1 - int(self._clip_at_start)]
        self._update_canvas()

    @do_not_record
    def clip_here(self):
        parent = self._parent_widget()
        parent.clip_spline(self._get_spline_id(), self.current_clip_length)

    @clip_length.connect
    def _clip(self, val: nm):
        restored = self._spline.restore()
        length = restored.length()
        original_spline = restored.clip(*self._original_lims)
        lim0, lim1 = self._original_lims
        if self._clip_at_start:
            self._current_lims = (lim0 + val / length, self._current_lims[1])
        else:
            self._current_lims = (self._current_lims[0], lim1 - val / length)

        self._spline = original_spline.clip(*self._current_lims)

    @clip_length.connect
    def _update_canvas(self):
        if self._spline is None:
            return None
        parent = self._parent_widget()
        tomo = parent.tomogram
        spl = self._spline
        binsize: int = parent.layer_image.metadata["current_binsize"]
        imgb = parent.layer_image.data

        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)

        # sample subtomogram at the edge
        mole = spl.anchors_to_molecules([0.0, 1.0])
        index = 0 if self._clip_at_start else -1
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
        self.xy_canvas.image = self._subtomograms.proj("z")
        self.yz_canvas.image = self._subtomograms.proj("x")
        self.xz_canvas.image = self._subtomograms.proj("y")

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
