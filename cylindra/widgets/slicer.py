from typing import TYPE_CHECKING, Callable
from magicclass import magicclass, field, vfield, MagicTemplate, bind_key, nogui
from magicclass.types import Optional, OneOf
from magicclass.ext.pyqtgraph import QtImageCanvas
import impy as ip

from cylindra.utils import map_coordinates
from cylindra.const import GlobalVariables as GVar, nm

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget
    from magicclass.fields import MagicValueField

    _FilterField = MagicValueField[Callable[[ip.ImgArray], ip.ImgArray]]

YPROJ = "Y-projection"
RPROJ = "R-projection"
CFT = "CFT"

POST_FILTERS: list[tuple[str, Callable[[ip.ImgArray], ip.ImgArray]]] = [
    ("None", lambda x: x),
    ("Low-pass (cutoff = 0.1)", lambda x: x.lowpass_filter(0.1)),
    ("Low-pass (cutoff = 0.2)", lambda x: x.lowpass_filter(0.2)),
]


@magicclass(record=False)
class SplineSlicer(MagicTemplate):
    show_what = vfield(label="kind").with_choices([YPROJ, RPROJ, CFT])

    @magicclass(layout="horizontal")
    class params(MagicTemplate):
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

        def _get_available_binsize(self, widget=None) -> "list[int]":
            from .main import CylindraMainWidget

            try:
                parent = self.find_ancestor(CylindraMainWidget)
                return parent._get_available_binsize(widget)
            except Exception:
                return []

        depth = vfield(32.0, label="depth (nm)").with_options(min=1.0, max=200.0)
        binsize = vfield(record=False).with_choices(_get_available_binsize)

    radius = vfield(Optional[nm], label="Radius (nm)").with_options(
        text="Use spline radius", options={"max": 200.0}
    )
    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)

    @property
    def parent(self) -> "CylindraMainWidget":
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget)

    @magicclass(widget_type="frame")
    class controller(MagicTemplate):
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
            from .main import CylindraMainWidget

            try:
                parent = self.find_ancestor(CylindraMainWidget)
                return parent._get_splines(widget)
            except Exception:
                return []

        spline_id = vfield(label="Spline").with_choices(_get_spline_id)
        pos = field(nm, label="Position (nm)", widget_type="FloatSlider").with_options(
            max=0
        )

        @bind_key("Up")
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)

        @bind_key("Down")
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)

    def refresh_widget_state(self):
        """Refresh widget state."""
        tomo = self.parent.tomogram
        if tomo is None:
            return None
        self.parent.tomogram.splines
        parent = self.parent
        tomo = parent.tomogram
        self._spline_changed(self.controller.spline_id)
        self._update_canvas()
        return None

    post_filter: "_FilterField" = vfield(OneOf[POST_FILTERS], label="Filter")

    @controller.spline_id.connect
    def _spline_changed(self, idx: int):
        try:
            spl = self.parent.tomogram.splines[idx]
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

    def _show_global_r_proj(self):
        """Show radial projection of cylindrical image along current spline."""
        i = self.controller.spline_id
        polar = self.parent.tomogram.straighten_cylindric(i).proj("r")
        self.canvas.image = polar.value
        self.canvas.text_overlay.update(
            visible=True, text=f"{i}-global", color="magenta"
        )
        return None

    def _show_global_ft(self, i: int):
        """View Fourier space along current spline."""
        polar = self.parent.tomogram.straighten_cylindric(i)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        pw[:] = pw / pw.max()

        self.canvas.image = pw.value
        self.canvas.text_overlay.update(
            visible=True, text=f"{i}-global", color="magenta"
        )
        return None

    def _current_cartesian_img(
        self, idx: int, pos: nm, depth: nm
    ) -> "ip.ImgArray | Exception":
        """Return local Cartesian image at the current position."""
        binsize = self.params.binsize
        if self.radius is None:
            hwidth = None
        else:
            hwidth = self.radius + GVar.thickness_outer
        try:
            return self.get_cartesian_image(
                idx, pos, depth=depth, binsize=binsize, half_width=hwidth
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
                idx, pos, depth=depth, binsize=binsize, radius=self.radius
            )
        except ValueError as e:
            return e

    @nogui
    def get_cartesian_image(
        self,
        spline: int,
        pos: nm,
        *,
        depth: nm = 32.0,
        binsize: int = 1,
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
        half_width : nm, optional
            Half width size of the image. (depth, 2 * half_width, 2 * half_width)
            will be the output image shape.

        Returns
        -------
        ip.ImgArray
            Cropped XYZ image.
        """
        tomo = self.parent.tomogram
        spl = tomo.splines[spline]
        depth_px = tomo.nm2pixel(depth, binsize=binsize)
        r = half_width or spl.radius + GVar.thickness_outer
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
        out = map_coordinates(img, coords, order=1)
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
        radius : nm, optional
            Radius peak of the cylinder.

        Returns
        -------
        ip.ImgArray
            Cylindric image.
        """
        tomo = self.parent.tomogram
        ylen = tomo.nm2pixel(depth, binsize=binsize)
        spl = tomo.splines[spline]

        r = radius or spl.radius
        if r is None:
            raise ValueError("Radius not available in the input spline.")
        rmin = tomo.nm2pixel(max(r - GVar.thickness_inner, 0), binsize=binsize)
        rmax = tomo.nm2pixel((r + GVar.thickness_outer), binsize=binsize)

        coords = spl.translate(
            [-tomo.multiscale_translation(binsize)] * 3
        ).local_cylindrical(
            r_range=(rmin, rmax),
            n_pixels=ylen,
            u=pos / spl.length(),
            scale=tomo.scale * binsize,
        )
        img = tomo._get_multiscale_or_original(binsize)
        polar = map_coordinates(img, coords, order=1)
        polar = ip.asarray(polar, axes="rya")  # radius, y, angle
        polar.set_scale(
            r=img.scale.x, y=img.scale.x, a=img.scale.x, unit=img.scale_unit
        )
        return polar
