from typing import Callable
from magicclass import (
    abstractapi,
    magicclass,
    field,
    set_design,
    vfield,
    bind_key,
    nogui,
)
from magicclass.logging import getLogger
from magicclass.types import Optional, OneOf
from magicclass.ext.pyqtgraph import QtImageCanvas
import impy as ip

from cylindra.utils import map_coordinates
from cylindra.const import nm
from cylindra.components._ftprops import LatticeAnalyzer, get_polar_image
from ._child_widget import ChildWidget

YPROJ = "Y-projection"
RPROJ = "R-projection"
CFT = "CFT"

POST_FILTERS: list[tuple[str, Callable[[ip.ImgArray], ip.ImgArray]]] = [
    ("None", lambda x: x),
    ("Low-pass (cutoff = 0.2)", lambda x: x.lowpass_filter(0.2)),
    ("Low-pass (cutoff = 0.1)", lambda x: x.lowpass_filter(0.1)),
]

_Logger = getLogger("cylindra")


@magicclass(record=False)
class SplineSlicer(ChildWidget):
    show_what = vfield(label="kind").with_choices([YPROJ, RPROJ, CFT])

    @magicclass(layout="horizontal")
    class params(ChildWidget):
        """
        Sweeper parameters.

        Attributes
        ----------
        depth : float
            The depth of the projection along splines. For instance, depth=32.0 means that Y-projection will be calculated
            using subvolume of size L * 32.0 nm * L.
        binsize : int
            The size of the binning. For instance, binsize=2 means that the image will be binned by 2 before projection
            and/or Fourier transformation.
        """

        def __init__(self):
            self._old_binsize = 1

        def _get_available_binsize(self, widget=None) -> "list[int]":
            try:
                return self._get_main()._get_available_binsize(widget)
            except Exception:
                return []

        depth = vfield(32.0, label="depth (nm)").with_options(min=1.0, max=200.0)
        binsize = vfield(record=False).with_choices(_get_available_binsize)

    @magicclass(layout="horizontal")
    class Row0(ChildWidget):
        radius = abstractapi()
        post_filter = abstractapi()

    radius = Row0.vfield(Optional[nm], label="Radius (nm)").with_options(
        text="Use spline global radius", options={"max": 200.0}
    )
    post_filter = Row0.vfield(OneOf[POST_FILTERS], label="Filter")
    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)

    @magicclass(widget_type="frame")
    class controller(ChildWidget):
        """
        Control spline positions.

        Attributes
        ----------
        spline_id : int
            Current spline ID to analyze.
        pos : nm
            Position along the spline in nm.
        """

        def _get_spline_id(self, widget=None) -> "list[tuple[str, int]]":
            try:
                return self._get_main()._get_splines(widget)
            except Exception:
                return []

        spline_id = vfield(label="Spline").with_choices(_get_spline_id)
        pos = field(nm, label="Position (nm)", widget_type="FloatSlider").with_options(max=0)  # fmt: skip

        @bind_key("Up")
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)

        @bind_key("Down")
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)

    @set_design(text="Refresh")
    def refresh_widget_state(self):
        """Refresh widget state."""
        main = self._get_main()
        tomo = main.tomogram
        if tomo is None:
            return None
        self._spline_changed(self.controller.spline_id)
        self._update_canvas()
        return None

    @set_design(text="Measure CFT")
    def measure_cft_here(self):
        """Measure CFT parameters at the current position."""
        idx, pos, depth = self._get_cropping_params()
        binsize = self.params.binsize
        radius = self._get_radius()
        img = self.get_cylindric_image(
            idx, pos, depth=depth, binsize=binsize, radius=radius, order=3
        )
        img = img - float(img.mean())
        tomo = self._get_main().tomogram
        spl = tomo.splines[idx]
        analyzer = LatticeAnalyzer(spl.config)
        params = analyzer.polar_ft_params(img, radius)
        spl_info = _col(f"spline ID = {idx}; position = {pos:.2f} nm", color="#003FFF")
        _Logger.print_html(
            f"{spl_info}<br>"
            f"{_col('spacing:')} {params.spacing:.3f} nm<br>"
            f"{_col('rise angle:')} {params.rise_angle:.3f} °<br>"
            f"{_col('rise length:')} {params.rise_length:.3f} nm<br>"
            f"{_col('skew angle:')} {params.skew_angle:.3f} °<br>"
            f"{_col('skew tilt angle:')} {params.skew_tilt:.3f} °<br>"
            f"{_col('PF:')} {params.npf}<br>"
            f"{_col('start:')} {params.start:.3f}"
        )

    def _get_cropping_params(self) -> tuple[int, nm, nm]:
        idx = self.controller.spline_id
        if idx is None:
            return self._show_overlay_text("No spline exists.")
        depth = self.params.depth
        pos = self.controller.pos.value
        return idx, pos, depth

    @controller.spline_id.connect
    def _spline_changed(self, idx: int):
        try:
            spl = self._get_main().tomogram.splines[idx]
            self.controller.pos.max = max(spl.length(), 0)
        except Exception:
            pass

    @show_what.connect
    @params.depth.connect
    @params.binsize.connect
    @radius.connect
    @controller.spline_id.connect
    @controller.pos.connect
    @post_filter.connect
    def _on_widget_state_changed(self):
        if self.visible:
            self._update_canvas()
            self.params._old_binsize = self.params.binsize
        return None

    def _update_canvas(self):
        _type = self.show_what
        idx = self.controller.spline_id
        if idx is None:
            return self._show_overlay_text("No spline exists.")
        depth = self.params.depth
        pos = self.controller.pos.value
        if _type == RPROJ:
            result = self._current_cylindrical_img(idx, pos, depth)
            if isinstance(result, Exception):
                return self._show_overlay_text(result)
            img = self.post_filter(result.proj("r")).value
        elif _type == YPROJ:
            result = self._current_cartesian_img(idx, pos, depth).proj("y")
            if isinstance(result, Exception):
                return self._show_overlay_text(result)
            img = self.post_filter(result[ip.slicer.x[::-1]]).value
        elif _type == CFT:
            result = self.post_filter(self._current_cylindrical_img(idx, pos, depth))
            if isinstance(result, Exception):
                return self._show_overlay_text(result)
            pw = result.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw[:] = pw / pw.max()
            img = pw.value
        else:
            raise RuntimeError
        self.canvas.image = img
        self.canvas.text_overlay.visible = False
        factor = self.params._old_binsize / self.params.binsize
        if factor != 1:
            xlim = [(v + 0.5) * factor - 0.5 for v in self.canvas.xlim]
            ylim = [(v + 0.5) * factor - 0.5 for v in self.canvas.ylim]
            self.canvas.xlim = xlim
            self.canvas.ylim = ylim
        return None

    def _show_overlay_text(self, txt):
        self.canvas.text_overlay.visible = True
        self.canvas.text_overlay.text = str(txt)
        self.canvas.text_overlay.anchor = (0, 0)
        self.canvas.text_overlay.color = "yellow"
        self.canvas.text_overlay.font_size = 20
        del self.canvas.image
        return

    @show_what.connect
    def _update_clims(self):
        img = self.canvas.image
        if img is not None:
            self.canvas.contrast_limits = (img.min(), img.max())
        return None

    def _current_cartesian_img(
        self, idx: int, pos: nm, depth: nm
    ) -> "ip.ImgArray | Exception":
        """Return local Cartesian image at the current position."""
        binsize = self.params.binsize
        if self.radius is None:
            hwidth = None
        else:
            cfg = self._get_main().tomogram.splines[idx].config
            hwidth = self.radius + cfg.thickness_outer
        try:
            return self.get_cartesian_image(
                idx,
                pos,
                depth=depth,
                binsize=binsize,
                half_width=hwidth,
                order=1,
            )
        except ValueError as e:
            return e

    def _current_cylindrical_img(
        self, idx: int, pos: nm, depth: nm
    ) -> "ip.ImgArray | Exception":
        """Return cylindric-transformed image at the current position"""
        binsize = self.params.binsize
        try:
            return self.get_cylindric_image(
                idx,
                pos,
                depth=depth,
                binsize=binsize,
                radius=self.radius,
                order=1,
            )
        except ValueError as e:
            return e

    def _get_radius(self) -> nm:
        if self.radius is None:
            idx = self.controller.spline_id
            if idx is None:
                raise ValueError("No spline exists.")
            spl = self._get_main().tomogram.splines[idx]
            return spl.radius
        return self.radius

    @nogui
    def get_cartesian_image(
        self,
        spline: int,
        pos: nm,
        *,
        depth: nm = 32.0,
        binsize: int = 1,
        order: int = 3,
        half_width: nm = None,
    ) -> ip.ImgArray:
        """
        Get XYZ-coordinated image along a spline.

        Parameters
        ----------
        spline : int
            The spline index.
        pos : nm
            Position of the center of the image. `pos` nm from the spline start
            point will be used.
        depth : nm, default is 32.0
            Depth of the output image. Depth corresponds to the length of the
            direction parallel to the spline vector at the given position.
        binsize : int, default is 1
            Image bin size to use.
        order : int, default is 3
            Interpolation order.
        half_width : nm, optional
            Half width size of the image. (depth, 2 * half_width, 2 * half_width)
            will be the output image shape.

        Returns
        -------
        ip.ImgArray
            Cropped XYZ image.
        """
        tomo = self._get_main().tomogram
        spl = tomo.splines[spline]
        depth_px = tomo.nm2pixel(depth, binsize=binsize)
        r = half_width or spl.radius + spl.config.thickness_outer
        width_px = tomo.nm2pixel(2 * r, binsize=binsize) + 1
        if r is None:
            raise ValueError("Measure spline radius or manually set it.")
        coords = spl.translate(
            [-tomo.multiscale_translation(binsize)] * 3
        ).local_cartesian(
            shape=(width_px, width_px),
            n_pixels=depth_px,
            u=pos / spl.length(),
            scale=tomo.scale * binsize,
        )
        img = tomo._get_multiscale_or_original(binsize)
        out = map_coordinates(img, coords, order=order)
        out = ip.asarray(out, axes="zyx")
        out.set_scale(img, unit=img.scale_unit)
        return out

    @nogui
    def get_cylindric_image(
        self,
        spline: int,
        pos: nm,
        *,
        depth: nm = 32.0,
        binsize: int = 1,
        order: int = 3,
        radius: nm = None,
    ) -> ip.ImgArray:
        """
        Get RYΘ-coordinated cylindric image.

        Parameters
        ----------
        spline : int
            The spline index.
        pos : nm
            Position of the center of the image. `pos` nm from the spline start
            point will be used.
        depth : nm, default is 32.0
            Depth of the output image. Depth corresponds to the length of the
            direction parallel to the spline vector at the given position.
        binsize : int, default is 1
            Image bin size to use.
        order : int, default is 3
            Interpolation order.
        radius : nm, optional
            Radius peak of the cylinder.

        Returns
        -------
        ip.ImgArray
            Cylindric image.
        """
        tomo = self._get_main().tomogram
        ylen = tomo.nm2pixel(depth, binsize=binsize)
        spl = tomo.splines[spline]

        r = radius or spl.radius
        if r is None:
            raise ValueError("Radius not available in the input spline.")
        _scale = tomo.scale * binsize
        rmin = max(r - spl.config.thickness_inner, 0) / _scale
        rmax = (r + spl.config.thickness_outer) / _scale

        coords = spl.translate(
            [-tomo.multiscale_translation(binsize)] * 3
        ).local_cylindrical(
            r_range=(rmin, rmax),
            n_pixels=ylen,
            u=pos / spl.length(),
            scale=tomo.scale * binsize,
        )
        img = tomo._get_multiscale_or_original(binsize)
        return get_polar_image(
            img, coords, radius=(rmin + rmax) / 2 * _scale, order=order
        )

    @bind_key("Esc")
    def _close_this_window(self):
        return self.close()


def _col(txt: str, color: str = "#FFFF00") -> str:
    return f'<b><font color="{color}">{txt}</font></b>'
