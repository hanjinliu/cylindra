from typing import TYPE_CHECKING, Any
import time
import numpy as np
import impy as ip

from magicclass import (
    abstractapi,
    magicclass,
    MagicTemplate,
    field,
    set_design,
    vfield,
    box,
)
from magicclass.logging import getLogger
from magicclass.types import OneOf, Path
from magicclass.utils import thread_worker
from magicclass.ext.pyqtgraph import QtMultiImageCanvas
from dask import delayed, array as da

from cylindra.const import PropertyNames as H, Mode
from cylindra.utils import map_coordinates, Projections
from cylindra.widgets.widget_utils import FileFilter
from ._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components import CylSpline
_Logger = getLogger("cylindra")


@delayed
def delayed_map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: np.ndarray,
):
    """Try to map coordinates. If failed (out-of-bound), return an zero array."""
    try:
        out = map_coordinates(
            input, coordinates, order=1, mode=Mode.constant, cval=np.mean
        )
    except ValueError:
        out = np.zeros(coordinates.shape[1:], dtype=input.dtype)
    return out


@magicclass(
    widget_type="groupbox", name="Spline Control", error_mode="stderr", record=False
)
class SplineControl(ChildWidget):
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
        self._projections = list[Projections]()

        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"

        self.canvas.enabled = False

    def _get_splines(self, *_) -> list[int]:
        """Get list of spline objects for categorical widgets."""
        try:
            tomo = self._get_main().tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    num = vfield(OneOf[_get_splines], label="Spline No.")
    pos = vfield(int, widget_type="Slider", label="Position").with_options(max=0)
    canvas = box.resizable(
        field(QtMultiImageCanvas, name="Figure").with_options(nrows=1, ncols=3),
        x_enabled=False,
    )

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)}, record=False)
    class footer(MagicTemplate):
        highlight_subvolume = vfield(False).with_options(text="Highlight subvolume")
        copy_screenshot = abstractapi()
        save_screenshot = abstractapi()
        log_screenshot = abstractapi()

    @footer.wraps
    @set_design(max_width=40, text="Copy")
    def copy_screenshot(self):
        """Copy a screenshot of the projections to clipboard."""
        return self.canvas.to_clipboard()

    @footer.wraps
    @set_design(max_width=40, text="Scr")
    def save_screenshot(self, path: Path.Save[FileFilter.PNG]):
        """Take a screenshot of the projections."""
        from skimage.io import imsave

        img = self.canvas.render()
        return imsave(path, img)

    @footer.wraps
    @set_design(max_width=40, text="Log")
    def log_screenshot(self):
        """Take a screenshot of the projections and show in the logger."""
        import matplotlib.pyplot as plt

        img = self.canvas.render()
        with _Logger.set_plt():
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return None

    @num.connect
    @pos.connect
    @footer.highlight_subvolume.connect
    def _highlight(self):
        """Change camera focus to the position of current spline fragment."""
        parent = self._get_main()
        if parent.parent_viewer is None:
            return None
        highlight = parent._reserved_layers.highlight
        if not self.footer.highlight_subvolume:
            if highlight in parent.parent_viewer.layers:
                parent.parent_viewer.layers.remove(highlight)
            return None

        tomo = parent.tomogram
        if len(tomo.splines) == 0:
            return None
        if highlight not in parent.parent_viewer.layers:
            parent.parent_viewer.add_layer(highlight)

        spl = tomo.splines[self.num]
        anc = spl.anchors[self.pos]
        highlight.data = spl.map(anc)
        scale = parent._reserved_layers.scale
        highlight.size = tomo.splines[self.num].config.fit_width / scale * 2

        return None

    @property
    def need_resample(self) -> bool:
        """True if the canvas is showing the old data."""
        return self.canvas[0].image is not None

    @num.connect
    def _update_pos_max(self, i: int | None):
        if i is None:
            return
        tomo = self._get_main().tomogram
        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]

        if len(spl.localprops) > 0:
            self["pos"].max = len(spl.localprops) - 1
        elif spl.has_anchors:
            self["pos"].max = spl.anchors.size - 1
        else:
            self.pos = 0
            self["pos"].max = 0
            return

    @num.connect_async(timeout=0.1)
    def _num_changed(self):
        i = self.num
        if i is None:
            return
        parent = self._get_main()
        tomo = parent.tomogram
        if i >= len(tomo.splines):
            return
        spl = tomo.splines[i]
        if len(spl.localprops) == 0:

            @thread_worker.callback
            def _on_yield():
                parent.LocalProperties._init_text()
                parent.LocalProperties._init_plot()

            yield _on_yield

            if not spl.has_anchors:
                return

        yield from self._load_projection(spl)
        yield  # breakpoint
        yield from self._update_canvas.arun()
        return None

    def _load_projection(self, spl: "CylSpline"):
        parent = self._get_main()
        tomo = parent.tomogram

        # update plots in pyqtgraph, if properties exist
        cb = thread_worker.callback(parent.LocalProperties._plot_properties)
        yield cb.with_args(spl)

        # calculate projection
        if (npfs := spl.props.get_loc(H.npf, None)) is not None:
            npf_list = npfs
        elif (npf := spl.props.get_glob(H.npf, None)) is not None:
            npf_list = [npf] * spl.anchors.size
        else:
            npf_list = [0] * spl.anchors.size

        binsize = parent._current_binsize
        imgb = parent.tomogram.get_multiscale(binsize)

        length_px = tomo.nm2pixel(spl.config.fit_depth, binsize=binsize)
        width_px = tomo.nm2pixel(spl.config.fit_width, binsize=binsize)

        mole = spl.anchors_to_molecules()
        if binsize > 1:
            mole = mole.translate(-parent.tomogram.multiscale_translation(binsize))
        loc_shape = (width_px, length_px, width_px)
        coords = mole.local_coordinates(
            shape=loc_shape,
            scale=tomo.scale * binsize,
        )
        yield
        projections = list[Projections]()
        for crds, npf in zip(coords, npf_list):
            mapped = delayed_map_coordinates(imgb, crds)
            vol = ip.LazyImgArray(
                da.from_delayed(mapped, shape=loc_shape, dtype=imgb.dtype), axes="zyx"
            )
            projections.append(Projections(vol, npf=npf))

        self._projections = projections
        return None

    @pos.connect_async(timeout=0.1, abort_limit=0.5)
    def _update_canvas(self):
        parent = self._get_main()
        tomo = parent.tomogram
        binsize = parent._current_binsize
        i = self.num
        j = self.pos

        if not self._projections or i is None or j is None:
            return self._clear_all_layers
        if i >= len(tomo.splines):
            return

        spl = tomo.splines[i]
        # Set projections
        proj = self._projections[j].compute()
        yield self._update_projections.with_args(proj)

        # Update text overlay
        if spl.has_anchors and j < len(spl.anchors):
            len_nm = f"{spl.length(0, spl.anchors[j]):.2f}"
        else:
            len_nm = "NA"

        yield self._update_text_overlay.with_args(f"{i}-{j} ({len_nm} nm)")

        if spl.props.has_loc(H.radius):
            radii = spl.props.get_loc(H.radius)
            if len(radii) != self["pos"].max + 1:
                return None
            r0 = radii[j]
        elif spl.props.has_glob(H.radius):
            r0 = spl.props.get_glob(H.radius)
        else:
            return None
        lz, ly, lx = np.array(proj.shape)

        depths = list(spl.props.window_size.values())
        if len(depths) == 0:
            ylen = 25 / binsize / tomo.scale
        else:
            ylen = depths[0] / 2 / binsize / tomo.scale

        # innter/outer radius of the cylinder
        r_inner = max(r0 - spl.config.thickness_inner, 0) / tomo.scale / binsize
        r_outer = (r0 + spl.config.thickness_outer) / tomo.scale / binsize

        # draw a square in YX-view
        ymin, ymax = ly / 2 - ylen - 0.5, ly / 2 + ylen + 0.5
        xmin, xmax = -r_outer + lx / 2 - 1, r_outer + lx / 2
        xy = [xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin]
        yield self._add_curve.with_args(0, [xy])

        # draw two circles in ZX-view
        center = (lx / 2 - 0.5, lz / 2 - 0.5)
        yield self._add_curve.with_args(
            1, [_circle(_r, center=center) for _r in [r_inner, r_outer]]
        )

        # draw polarity

        @thread_worker.callback
        def _on_return():
            kw = dict(size=16, color="lime", anchor=(0.5, 0.5))
            if spl.orientation == "PlusToMinus":
                self.canvas[1].add_text(*center, "+", **kw)
            elif spl.orientation == "MinusToPlus":
                self.canvas[1].add_text(*center, "-", **kw)

            # update pyqtgraph
            if (xs := spl.props.get_loc(H.spl_dist, None)) is not None:
                parent.LocalProperties._plot_spline_position(xs[j])
            else:
                parent.LocalProperties._init_plot()

        return _on_return

    @thread_worker.callback
    def _clear_all_layers(self):
        for ic in range(3):
            self.canvas[ic].layers.clear()

    @thread_worker.callback
    def _update_projections(self, proj: Projections):
        self.canvas[0].image = proj.yx
        self.canvas[1].image = proj.zx
        if proj.zx_ave is not None:
            self.canvas[2].image = proj.zx_ave
        else:
            del self.canvas[2].image

    @thread_worker.callback
    def _update_text_overlay(self, txt: str):
        self.canvas[0].text_overlay.text = txt
        self.canvas[0].text_overlay.color = "lime"

    @thread_worker.callback
    def _add_curve(self, i: int, data: list[tuple[Any, Any]]):
        self.canvas[i].layers.clear()
        for x, y in data:
            self.canvas[i].add_curve(x, y, color="lime", antialias=True)

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
