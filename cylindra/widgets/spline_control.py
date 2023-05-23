import numpy as np
import impy as ip

from magicclass import (
    magicclass,
    MagicTemplate,
    field,
    vfield,
)
from magicclass.types import OneOf
from magicclass.ext.pyqtgraph import QtMultiImageCanvas

from cylindra.const import GlobalVariables as GVar, PropertyNames as H, Mode
from cylindra.utils import map_coordinates, Projections


@magicclass(widget_type="groupbox", name="Spline Control")
class SplineControl(MagicTemplate):
    """
    Control and visualization along splines

    Attributes
    ----------
    num : int
        Splines in current tomogram.
    pos : int
        Position along the spline.
    canvas : QtMultiImageCanvas
        2-D projections of subtomogram at current position.
    """

    def __post_init__(self):
        self.projections: list[Projections] = []

        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"

        self.canvas.enabled = False

    def _get_parent(self):
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    def _get_splines(self, *_) -> list[int]:
        """Get list of spline objects for categorical widgets."""
        try:
            tomo = self._get_parent().tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    num = vfield(OneOf[_get_splines], label="Spline No.", record=False)
    pos = vfield(
        int, widget_type="Slider", label="Position", record=False
    ).with_options(max=0)
    canvas = field(QtMultiImageCanvas, name="Figure").with_options(nrows=1, ncols=3)

    @magicclass(layout="horizontal")
    class footer(MagicTemplate):
        focus = vfield(False, record=False).with_options(
            text="focus on",
            tooltip="Keep focus of viewer camera on the current spline position",
        )

    @num.connect
    @pos.connect
    @footer.focus.connect
    def _highlight(self):
        """Change camera focus to the position of current spline fragment."""
        parent = self._get_parent()
        layer = parent.layer_paint
        if layer is None:
            return None

        # NOTE: the setter of "show_selected_label" calls layer.refresh() so that
        # it is very slow. Check if "show_selected_label" is True before setting it.
        if not self.footer.focus:
            if layer.show_selected_label:
                layer.show_selected_label = False
            return None

        i = self.num
        j = self.pos

        tomo = parent.tomogram

        if not layer.show_selected_label:
            layer.show_selected_label = True

        j_offset = sum(spl.anchors.size for spl in tomo.splines[:i])
        layer.selected_label = j_offset + j + 1
        return None

    @property
    def need_resample(self) -> bool:
        """True if the canvas is showing the old data."""
        return self.canvas[0].image is not None

    @num.connect
    def _num_changed(self):
        from cylindra.widgets.main import CylindraMainWidget

        i = self.num
        if i is None:
            return
        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]

        if len(spl.localprops) > 0:
            n_anc = len(spl.localprops)
        else:
            parent.LocalProperties._init_text()
            parent.LocalProperties._init_plot()
            if spl._anchors is not None:
                n_anc = len(spl._anchors)
            else:
                self.pos = 0
                self["pos"].max = 0
                return

        self["pos"].max = n_anc - 1
        self._load_projection()
        self._update_canvas()
        return None

    def _load_projection(self):
        from cylindra.widgets.main import CylindraMainWidget

        i = self.num
        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]

        # update plots in pyqtgraph, if properties exist
        parent.LocalProperties._plot_properties(spl)

        # calculate projection
        if (npfs := spl.get_localprops(H.nPF, None)) is not None:
            npf_list = npfs
        elif (npf := spl.get_globalprops(H.nPF, None)) is not None:
            npf_list = [npf] * spl.anchors.size
        else:
            npf_list = [0] * spl.anchors.size

        binsize = parent.layer_image.metadata["current_binsize"]
        imgb = parent.tomogram.get_multiscale(binsize)

        length_px = tomo.nm2pixel(GVar.fit_depth, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fit_width, binsize=binsize)

        mole = spl.anchors_to_molecules()
        if binsize > 1:
            mole = mole.translate(-parent.tomogram.multiscale_translation(binsize))
        coords = mole.local_coordinates(
            shape=(width_px, length_px, width_px),
            scale=tomo.scale * binsize,
        )
        out: list[ip.ImgArray] = []
        for crds in coords:
            mapped = map_coordinates(
                imgb, crds, order=1, mode=Mode.constant, cval=np.mean
            )
            out.append(ip.asarray(mapped, axes="zyx"))

        projections = []
        for img, npf in zip(out, npf_list):
            proj = Projections(img)
            if npf > 1:
                proj.rotational_average(npf)
            projections.append(proj)

        self.projections = projections
        return None

    @pos.connect
    def _update_canvas(self):
        from .main import CylindraMainWidget

        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        binsize = parent.layer_image.metadata["current_binsize"]
        i = self.num
        j = self.pos
        if i >= len(tomo.splines):
            return
        if not self.projections or i is None or j is None:
            for ic in range(3):
                self.canvas[ic].layers.clear()
            return
        spl = tomo.splines[i]
        # Set projections
        proj = self.projections[j]
        for ic in range(3):
            self.canvas[ic].layers.clear()
        self.canvas[0].image = proj.yx
        self.canvas[1].image = proj.zx
        if proj.zx_ave is not None:
            self.canvas[2].image = proj.zx_ave
        else:
            del self.canvas[2].image

        # Update text overlay
        self.canvas[0].text_overlay.text = f"{i}-{j}"
        self.canvas[0].text_overlay.color = "lime"

        if spl.radius is None:
            return None
        lz, ly, lx = np.array(proj.shape)

        if parent._current_ft_size is None:
            ylen = 25 / binsize / tomo.scale
        else:
            ylen = parent._current_ft_size / 2 / binsize / tomo.scale

        # draw a square in YX-view
        ymin, ymax = ly / 2 - ylen - 0.5, ly / 2 + ylen + 0.5
        r_inner = max(spl.radius - GVar.thickness_inner, 0) / tomo.scale / binsize
        r_outer = (spl.radius + GVar.thickness_outer) / tomo.scale / binsize
        xmin, xmax = -r_outer + lx / 2 - 1, r_outer + lx / 2
        self.canvas[0].add_curve(
            [xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime"
        )

        # draw two circles in ZX-view
        center = (lx / 2 - 0.5, lz / 2 - 0.5)
        self.canvas[1].add_curve(*_circle(r_inner, center=center), color="lime")
        self.canvas[1].add_curve(*_circle(r_outer, center=center), color="lime")

        # draw polarity
        kw = dict(size=16, color="lime", anchor=(0.5, 0.5))
        if spl.orientation == "PlusToMinus":
            self.canvas[2].add_text(*center, "+", **kw)
        elif spl.orientation == "MinusToPlus":
            self.canvas[2].add_text(*center, "-", **kw)

        # update pyqtgraph
        if (xs := spl.get_localprops(H.splDist, None)) is not None:
            parent.LocalProperties._plot_spline_position(xs[j])
        else:
            parent.LocalProperties._init_plot()

    def _reset_contrast_limits(self):
        for i in range(3):
            img = self.canvas[i].image
            if img is not None:
                self.canvas[i].contrast_limits = [img.min(), img.max()]
        return None


def _circle(
    r: float, center: tuple[float, float] = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Return the coordinates of a circle with radius r and center center."""
    theta = np.linspace(0, 2 * np.pi, 360)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y
