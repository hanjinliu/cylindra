from contextlib import suppress
from typing import Callable

import impy as ip
import numpy as np
from magicclass import (
    abstractapi,
    bind_key,
    field,
    magicclass,
    nogui,
    set_design,
    vfield,
)
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.logging import getLogger
from magicclass.types import Optional
from magicclass.utils import thread_worker
from numpy.typing import NDArray

from cylindra.components._ftprops import LatticeAnalyzer
from cylindra.const import nm
from cylindra.cyltransform import get_polar_image
from cylindra.utils import map_coordinates
from cylindra.widgets.subwidgets._child_widget import ChildWidget

YPROJ = "Y-projection"
RPROJ = "R-projection"
RPROJ_FILT = "Filtered-R-projection"
CFT = "CFT"
CFT_UP = "CFT (5x upsampling)"

_CHOICES = [YPROJ, RPROJ, RPROJ_FILT, CFT, CFT_UP]

POST_FILTERS: list[tuple[str, Callable[[ip.ImgArray], ip.ImgArray]]] = [
    ("None", lambda x: x),
    ("Low-pass (cutoff = 0.2)", lambda x: x.lowpass_filter(0.2)),
    ("Low-pass (cutoff = 0.1)", lambda x: x.lowpass_filter(0.1)),
]

_Logger = getLogger("cylindra")


@magicclass(record=False)
class SplineSlicer(ChildWidget):
    """Slicer along spline.

    Attributes
    ----------
    show_what : str
        Specifies what type of slice image will be shown in the canvas.
    radius : nm
        Radius of the cylinder. This value is used to restrict the xy size of the
        Y-projection and build cylindrical coordinate system. Changing this value does
        not overwrite the "radius" property of the spline. To overwrite, please use the
        "Apply radius and thickness" method.
    post_filter : str
        Post-filtering method to denoise the sliced image. This is usually used for
        denoising the Y-projection images.
    thickness_inner : nm
        Thickness of the cylinder inside the radius.
    thickness_outer : nm
        Thickness of the cylinder outside the radius.
    """

    show_what = vfield(label="kind").with_choices(_CHOICES)

    def __init__(self):
        self._current_cparams = None

    @magicclass(layout="horizontal")
    class params(ChildWidget):
        """Slicing parameters.

        Attributes
        ----------
        depth : float
            The depth of the projection along splines. For instance, depth=50.0 means
            that Y-projection will be calculated using subvolume of size L * 50.0 nm * L.
        binsize : int
            The size of the binning. For instance, binsize=2 means that the image will
            be binned by 2 before projection and/or Fourier transformation.
        """

        def __init__(self):
            self._old_binsize = 1

        def _get_available_binsize(self, widget=None) -> "list[int]":
            try:
                return self._get_main()._get_available_binsize(widget)
            except Exception:
                return []

        depth = vfield(50.0, label="depth (nm)").with_options(min=1.0, max=200.0)
        binsize = vfield().with_choices(_get_available_binsize)

    @magicclass(layout="horizontal")
    class Row0(ChildWidget):
        radius = abstractapi()
        post_filter = abstractapi()

    @magicclass(layout="horizontal")
    class Row1(ChildWidget):
        thickness_inner = abstractapi()
        thickness_outer = abstractapi()

    radius = vfield(Optional[nm], label="Radius (nm)", location=Row0).with_options(
        text="Use spline global radius",
        options={"min": 1.0, "max": 200.0, "step": 0.5, "value": 10.0},
    )
    post_filter = vfield(label="Filter", location=Row0).with_choices(POST_FILTERS)
    thickness_inner = vfield(2.0, label="Inner thickness (nm)", location=Row1).with_options(min=0.1, max=10.0, step=0.1)  # fmt: skip
    thickness_outer = vfield(2.0, label="Outer thickness (nm)", location=Row1).with_options(min=0.1, max=10.0, step=0.1)  # fmt: skip
    canvas = field(QtImageCanvas).with_options(lock_contrast_limits=True)

    @magicclass(widget_type="frame")
    class controller(ChildWidget):
        """Control spline positions.

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

    @magicclass(layout="horizontal")
    class Row2(ChildWidget):
        fit_spline_manually = abstractapi()
        measure_radius = abstractapi()
        measure_cft_here = abstractapi()

    @magicclass(layout="horizontal")
    class Row3(ChildWidget):
        apply_radius_and_thickness = abstractapi()
        refresh_widget_state = abstractapi()

    def __post_init__(self):
        self._circ_inner = self.canvas.add_curve(
            [], [], color="lime", lw=2, ls="--", antialias=True
        )
        self._circ_outer = self.canvas.add_curve(
            [], [], color="lime", lw=2, ls="--", antialias=True
        )

    def _update_circles(self, radius: float | None):
        if self.show_what != YPROJ or radius is None:
            self._circ_inner.visible = False
            self._circ_outer.visible = False
            return
        self._circ_inner.visible = True
        self._circ_outer.visible = True
        _scale = self._get_main().tomogram.scale * self.params.binsize
        r_inner = max(radius - self.thickness_inner, 0) / _scale
        r_outer = (radius + self.thickness_outer) / _scale

        theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        ny, nx = self.canvas.image.shape
        self._circ_inner.data = (
            r_inner * np.cos(theta) + (nx - 1) / 2,
            r_inner * np.sin(theta) + (ny - 1) / 2,
        )
        self._circ_outer.data = (
            r_outer * np.cos(theta) + (nx - 1) / 2,
            r_outer * np.sin(theta) + (ny - 1) / 2,
        )

    @bind_key("Up")
    @bind_key("F")
    def _next_pos(self):
        c = self.controller
        depth = self.params.depth
        c.pos.value = min(c.pos.value + depth, c.pos.max)

    @bind_key("Down")
    @bind_key("B")
    def _prev_pos(self):
        c = self.controller
        c.pos.value = max(c.pos.value - 1, c.pos.min)

    @set_design(text="Measure radius", location=Row2)
    def measure_radius(self):
        """Measure the radius for the current spline."""
        idx = self.controller.spline_id
        main = self._get_main()
        main.measure_radius([idx], bin_size=self.params.binsize)
        self.radius = main.tomogram.splines[idx].radius

    @set_design(text="Fit spline manually ...", location=Row2)
    def fit_spline_manually(self):
        """Open the spline fitter and run fitting tasks there."""
        main = self._get_main()
        main.SplinesMenu.Fitting.fit_splines_manually()
        main.spline_fitter.num.value = self.controller.spline_id

    @set_design(text="Measure CFT", location=Row2)
    def measure_cft_here(self):
        """Measure CFT parameters at the current position."""
        idx, pos, depth = self._get_cropping_params()
        binsize = self.params.binsize
        radius = self._get_radius()
        img = self.get_cylindric_image(
            idx, pos, depth=depth, binsize=binsize, radius=radius, order=3
        )
        tomo = self._get_main().tomogram
        spl = tomo.splines[idx]
        analyzer = LatticeAnalyzer(spl.config)
        rc = radius + (-spl.config.thickness_inner + self.thickness_outer) / 2
        params = analyzer.estimate_lattice_params_polar(img, rc)
        spl_info = _col(f"spline ID = {idx}; position = {pos:.2f} nm", color="#003FFF")
        _Logger.print_html(
            f"{spl_info}<br>"
            f"{_col('spacing:')} {params.spacing:.3f} nm<br>"
            f"{_col('rise angle:')} {params.rise_angle:.3f} °<br>"
            f"{_col('rise length:')} {params.rise_length:.3f} nm<br>"
            f"{_col('twist:')} {params.twist:.3f} °<br>"
            f"{_col('skew angle:')} {params.skew:.3f} °<br>"
            f"{_col('PF:')} {params.npf}<br>"
            f"{_col('start:')} {params.start}"
        )

    @set_design(text="Refresh", location=Row3)
    def refresh_widget_state(self):
        """Refresh widget state."""
        self._spline_changed(self.controller.spline_id, refer_config=False)
        return self._update_canvas()

    @set_design(text="Apply radius and thickness", location=Row3)
    def apply_radius_and_thickness(self):
        """Apply the current radius and thickness to the current spline."""
        idx = self.controller.spline_id
        main = self._get_main()
        main.update_spline_config(
            [idx],
            thickness_inner=self.thickness_inner,
            thickness_outer=self.thickness_outer,
        )  # fmt: skip
        if (r0 := self.radius) is not None:
            main.set_radius([idx], r0)
        self._spline_changed(idx, refer_config=False)

    def _get_cropping_params(self) -> tuple[int, nm, nm]:
        idx = self.controller.spline_id
        if idx is None:
            return self._show_overlay_text("No spline exists.")
        depth = self.params.depth
        pos = self.controller.pos.value
        return idx, pos, depth

    @controller.spline_id.connect
    def _spline_changed(self, idx: int, refer_config: bool = True):
        try:
            spl = self._get_main().tomogram.splines[idx]
            self.controller.pos.max = max(spl.length(), 0)
            if refer_config:
                self.thickness_inner = spl.config.thickness_inner
                self.thickness_outer = spl.config.thickness_outer
        except Exception:
            pass
        else:
            with suppress(Exception):
                self._current_cparams = spl.cylinder_params()

    @show_what.connect_async(timeout=0.1)
    @params.binsize.connect_async(timeout=0.1)
    def _on_show_what_changed(self):
        if self.visible:
            yield from self._update_canvas.arun(update_clim=True)
        return None

    @params.depth.connect_async(timeout=0.1)
    @radius.connect_async(timeout=0.1)
    @post_filter.connect_async(timeout=0.1)
    @controller.spline_id.connect_async(timeout=0.1)
    @controller.pos.connect_async(timeout=0.1, abort_limit=0.5)
    @thickness_inner.connect_async(timeout=0.1)
    @thickness_outer.connect_async(timeout=0.1)
    def _on_widget_state_changed(self):
        if self.visible:
            yield from self._update_canvas.arun()
        return None

    @thread_worker(force_async=True)
    def _update_canvas(self, update_clim: bool = False):
        _type = self.show_what
        idx = self.controller.spline_id
        if idx is None:
            return self._show_overlay_text_cb.with_args("No spline exists.")
        depth = self.params.depth
        pos = self.controller.pos.value
        if _type == RPROJ:
            result = self._current_cylindrical_img(idx, pos, depth)
            if isinstance(result, Exception):
                return self._show_overlay_text_cb.with_args(result)
            yield
            img2d = result.mean(axis="r")
            yield
            img = self.post_filter(img2d).value
        elif _type == YPROJ:
            result = self._current_cartesian_img(idx, pos, depth)
            if isinstance(result, Exception):
                return self._show_overlay_text_cb.with_args(result)
            yield
            img2d = result.mean(axis="y")[ip.slicer.x[::-1]]
            yield
            img = self.post_filter(img2d).value
        elif _type == CFT:
            result = self._current_cps(idx, pos, depth)
            if isinstance(result, Exception):
                return self._show_overlay_text_cb.with_args(result)
            yield
            result[:] = result / result.max()
            img = result.value
        elif _type == CFT_UP:
            result = self._current_cylindrical_img(idx, pos, depth)
            if isinstance(result, Exception):
                return self._show_overlay_text_cb.with_args(result)
            yield
            spl = self._get_main().tomogram.splines[idx]

            y0 = int(round(depth / spl.config.spacing_range.max * 1.2))
            a0 = int(round(spl.config.npf_range.max * 6))
            pw = result.local_power_spectra(
                f"y={-y0}:{y0+1};a={-a0}:{a0+1}", dims="rya", upsample_factor=[1, 5, 5]
            ).mean(axis="r")
            yield
            pw[:] = pw / pw.max()
            img = pw.value
        elif _type == RPROJ_FILT:
            result = self.post_filter(self._current_cylindrical_img(idx, pos, depth))
            if isinstance(result, Exception):
                return self._show_overlay_text_cb.with_args(result)
            yield
            ft = result.fft(shift=False, dims="rya")
            yield
            peaks = self._infer_peak_positions(ft)
            if isinstance(peaks, Exception):
                return self._show_overlay_text_cb.with_args(peaks)
            yield
            mask = _create_mask(ft.shape[1:], peaks)
            yield
            ft0 = ft[0] * mask
            img = ft0.ifft(shift=False, dims=ft0.axes).value
        else:
            raise RuntimeError(_type)

        yield

        @thread_worker.callback
        def _update_image():
            self.canvas.image = img
            self.canvas.text_overlay.visible = False
            factor = self.params._old_binsize / self.params.binsize
            if factor != 1:
                xlim = [(v + 0.5) * factor - 0.5 for v in self.canvas.xlim]
                ylim = [(v + 0.5) * factor - 0.5 for v in self.canvas.ylim]
                self.canvas.xlim = xlim
                self.canvas.ylim = ylim
            self.params._old_binsize = self.params.binsize
            self._update_circles(self._get_radius())

        yield _update_image

        if update_clim:
            lims = img.min(), img.max()

            @thread_worker.callback
            def _update_clim():
                self.canvas.contrast_limits = lims

            return _update_clim
        return None

    def _show_overlay_text(self, txt):
        self.canvas.text_overlay.visible = True
        self.canvas.text_overlay.text = str(txt)
        self.canvas.text_overlay.anchor = (0, 0)
        self.canvas.text_overlay.color = "yellow"
        self.canvas.text_overlay.font_size = 20
        del self.canvas.image
        return

    @thread_worker.callback
    def _show_overlay_text_cb(self, txt):
        return self._show_overlay_text(txt)

    def _current_cartesian_img(
        self, idx: int, pos: nm, depth: nm
    ) -> "ip.ImgArray | Exception":
        """Return local Cartesian image at the current position."""
        binsize = self.params.binsize
        if self.radius is None:
            hwidth = None
        else:
            hwidth = self.radius + self.thickness_outer * 2
        try:
            return self.get_cartesian_image(
                idx,
                pos,
                depth=depth,
                binsize=binsize,
                half_width=hwidth,
                order=1,
                use_orig_config=False,
            )
        except Exception as e:
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
                use_orig_config=False,
            )
        except Exception as e:
            return e

    def _current_cps(self, idx: int, pos: nm, depth: nm) -> "ip.ImgArray | Exception":
        """Return local power spectrum at the current position."""
        binsize = self.params.binsize
        try:
            return self.get_cylindric_power_spectrum(
                idx,
                pos,
                depth=depth,
                binsize=binsize,
                order=1,
                use_orig_config=False,
            )
        except Exception as e:
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
        depth: nm = 50.0,
        binsize: int = 1,
        order: int = 3,
        half_width: nm | None = None,
        use_orig_config: bool = True,
    ) -> ip.ImgArray:
        """Get XYZ-coordinated image along a spline.

        Parameters
        ----------
        spline : int
            The spline index.
        pos : nm
            Position of the center of the image. `pos` nm from the spline start
            point will be used.
        depth : nm, default 50.0
            Depth of the output image. Depth corresponds to the length of the
            direction parallel to the spline vector at the given position.
        binsize : int, default
            Image bin size to use.
        order : int, default 3
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
        if half_width is None and spl.radius is None:
            raise ValueError(
                "Radius not set to the spline. Measure spline radius using the "
                "'Measure radius' button or manually set it by 'Radius (nm)' entry."
            )
        if use_orig_config:
            r = half_width or spl.radius + spl.config.thickness_outer * 2
        else:
            r = half_width or spl.radius + self.thickness_outer * 2
        coords = spl.translate(
            [-tomo.multiscale_translation(binsize)] * 3
        ).local_cartesian(
            shape=(2 * r, 2 * r),
            depth=depth,
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
        depth: nm = 50.0,
        binsize: int = 1,
        order: int = 3,
        radius: nm | None = None,
        use_orig_config: bool = True,
    ) -> ip.ImgArray:
        """Get RYΘ-coordinated cylindric image.

        Parameters
        ----------
        spline : int
            The spline index.
        pos : nm
            Position of the center of the image. `pos` nm from the spline start
            point will be used.
        depth : nm, default 50.0
            Depth of the output image. Depth corresponds to the length of the
            direction parallel to the spline vector at the given position.
        binsize : int, default 1
            Image bin size to use.
        order : int, default 3
            Interpolation order.
        radius : nm, optional
            Radius peak of the cylinder.

        Returns
        -------
        ip.ImgArray
            Cylindric image.
        """
        tomo = self._get_main().tomogram
        spl = tomo.splines[spline]
        r = radius or spl.radius
        if r is None:
            raise ValueError("Radius not available in the input spline.")
        img = tomo._get_multiscale_or_original(binsize)
        _scale = img.scale.x
        rmin, rmax = spl.with_config(
            {
                "thickness_inner": self.thickness_inner,
                "thickness_outer": self.thickness_outer,
            },
            copy_props=True,
        ).radius_range()
        spl_trans = spl.translate([-tomo.multiscale_translation(binsize)] * 3)
        anc = pos / spl.length()
        coords = spl_trans.local_cylindrical((rmin, rmax), depth, anc, scale=_scale)
        return get_polar_image(img, coords, radius=(rmin + rmax) / 2, order=order)

    @nogui
    def get_cylindric_power_spectrum(
        self,
        spline: int,
        pos: nm,
        *,
        depth: nm = 50.0,
        binsize: int = 1,
        order: int = 3,
        use_orig_config: bool = True,
    ) -> ip.ImgArray:
        """Get cylindric power spectrum of given position.

        Parameters
        ----------
        spline : int
            The spline index.
        pos : nm
            Position of the center of the image. `pos` nm from the spline start
            point will be used.
        depth : nm, default 50.0
            Depth of the output image. Depth corresponds to the length of the
            direction parallel to the spline vector at the given position.
        binsize : int, default 1
            Image bin size to use.
        order : int, default 3
            Interpolation order.

        Returns
        -------
        ip.ImgArray
            Cylindric power spectrum.
        """
        result = self.get_cylindric_image(
            spline,
            pos,
            depth=depth,
            binsize=binsize,
            order=order,
            use_orig_config=use_orig_config,
        )
        pw = result.power_spectra(zero_norm=True, dims="rya").mean(axis="r")
        return pw

    @bind_key("Esc")
    def _close_this_window(self):
        return self.close()

    def _infer_peak_positions(
        self,
        img_cyl: ip.ImgArray,
        vsub: int = 2,
        hsub: int = 1,
        max_order: int = 1,
    ) -> "list[tuple[float, float]] | ValueError":
        cp = self._current_cparams
        if cp is None:
            return ValueError("No spline parameter can be used.")
        idx = self.controller.spline_id
        if idx is None:
            return ValueError("No spline exists.")
        ya_ratio = img_cyl.scale.y / img_cyl.scale.a
        vy = img_cyl.scale.y / cp.pitch
        vx = vy * cp.tan_rise_raw / ya_ratio
        hx = img_cyl.scale.a / cp.lat_spacing_proj
        hy = hx * cp.tan_skew / ya_ratio
        v = np.array([vy, vx]) / vsub
        h = np.array([hy, hx]) / hsub
        v0, h0 = max_order * vsub, max_order * hsub
        vmesh, hmesh = np.meshgrid(range(-v0, v0 + 1), range(-h0, h0 + 1))
        pos_all = vmesh.reshape(-1, 1) * v + hmesh.reshape(-1, 1) * h
        posv = pos_all[:, 0]
        posh = pos_all[:, 1]
        pos_valid = (
            (-0.5 <= posv)
            & (posv <= 0.5)
            & (-0.5 <= posh)
            & (posh <= 0.5)
            & (posv**2 + posh**2 > 0)
        )
        pos = pos_all[pos_valid]
        ny = img_cyl.shape.y
        na = img_cyl.shape.a
        return [(p[0] * ny, p[1] * na) for p in pos]


def _col(txt: str, color: str = "#FFFF00") -> str:
    return f'<b><font color="{color}">{txt}</font></b>'


def _create_mask(
    shape: tuple[int, int],
    positions: list[tuple[float, float]],
    sigma: float = 1.0,
) -> NDArray[np.float32]:
    placeholder = np.zeros(shape, dtype=np.float32)
    for each in positions:
        placeholder += _gaussian_2d(shape, each, sigma=sigma)
    return placeholder


def _gaussian_2d(
    shape: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 1.0,
) -> NDArray[np.float32]:
    y, a = np.indices(shape, dtype=np.float32)
    ny, na = shape
    yc, ac = center
    yc = yc + ny // 2
    ac = ac + na // 2
    y -= yc
    a -= ac
    arr = np.exp(-0.5 * (y**2 + a**2) / sigma**2)
    return np.fft.ifftshift(arr)
