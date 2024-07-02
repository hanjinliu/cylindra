from enum import Enum
from typing import TYPE_CHECKING, Annotated

import impy as ip
import numpy as np
import pyqtgraph as pg
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    get_button,
    magicclass,
    set_design,
    vfield,
)
from magicclass.ext.pyqtgraph import QtImageCanvas, mouse_event
from magicclass.types import Path
from qtpy.QtCore import Qt

from cylindra.const import FileFilter
from cylindra.const import PropertyNames as H
from cylindra.utils import roundint
from cylindra.widget_utils import capitalize
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components.tomogram._misc import ImageWithPeak


class MouseMode(Enum):
    """Define the behavior of the mouse click event."""

    none = "none"
    axial = "axial"
    angular = "angular"
    upsample = "upsample"


@magicclass(record=False)
class PeakInspector(ChildWidget):
    show_what = vfield(
        "Global-CFT", widget_type="RadioButtons", name="Show:"
    ).with_choices(["Local-CFT", "Global-CFT"])
    canvas = field(QtImageCanvas)
    pos = vfield(int, widget_type="Slider", label="Position").with_options(max=0)

    def _set_peaks(self, peaks: "list[ImageWithPeak]"):
        self._power_spectra = [peak.power for peak in peaks]
        self._peaks = peaks

    def reset_choices(self, *_):
        pass

    def __init__(self):
        self._peaks: "list[ImageWithPeak] | None" = None
        self._power_spectra = list[ip.ImgArray | None]()
        self._image = np.zeros((1, 1))
        self._is_log_scale = False
        self._current_spline_index = 0
        self._current_binsize = 1
        self._last_upsample_params = {}

    def __post_init__(self):
        self._upsampled_image_item = pg.ImageItem()
        self.canvas._viewbox.addItem(self._upsampled_image_item)
        self._upsampled_image_item.setVisible(False)
        self._infline_x = self.canvas.add_infline((0, 0), 0, color="yellow")
        self._infline_y = self.canvas.add_infline((0, 0), 90, color="yellow")
        self._layer_axial = self.canvas.add_scatter(
            [], [], color="cyan", symbol="+", size=12
        )
        self._layer_angular = self.canvas.add_scatter(
            [], [], color="lime", symbol="+", size=12
        )
        self._markers = self.canvas.add_scatter(
            [], [], color="red", symbol="+", size=15
        )
        self._texts = self.canvas.add_text(
            [0, 0],
            [0, 0],
            ["", ""],
            color="red",
            size=14,
        )

    def _set_spline(self, i: int, binsize: int | None = None):
        main = self._get_main()
        if i >= len(main.splines):
            # maybe initialized
            if len(main.splines) == 0:
                self._power_spectra = []
                self._peaks = None
                self._image = np.zeros((1, 1))
                return None
            else:
                raise ValueError(f"Invalid spline index: {i}")
        spl = main.splines[i]
        self._layer_axial.data = [], []
        self._layer_angular.data = [], []
        self._upsampled_image_item.setVisible(False)
        self._current_spline_index = i
        if self.show_what == "Local-CFT":
            if spl.props.has_loc(H.twist) and spl.has_anchors:
                # has local-CFT results
                if binsize is None:
                    binsize = spl.props.binsize_loc[H.twist]
                peaks = main.tomogram.local_image_with_peaks(i=i, binsize=binsize)
                self._set_peaks(peaks)
            else:
                # will not show anything
                self._power_spectra = [None]
                self._peaks = None
                binsize = 1
            self["pos"].visible = True
            self["pos"].max = len(self._power_spectra) - 1
        else:
            if self._may_show_text_overlay(
                spl.radius is None, "No CFT available (radius not set)"
            ):
                return
            if spl.props.has_glob(H.twist):
                # has global-CFT results
                if binsize is None:
                    binsize = spl.props.binsize_glob[H.twist]
                peak = [main.tomogram.global_image_with_peaks(i=i, binsize=binsize)]
                self._set_peaks(peak)
            else:
                # calculate global-CFT here (for manual measurement)
                if binsize is None:
                    if spl.props.has_glob(H.twist):
                        binsize = spl.props.binsize_glob[H.twist]
                    else:
                        main = self._get_main()
                        binsize = roundint(
                            main._reserved_layers.scale / main.tomogram.scale
                        )
                self._power_spectra = [
                    main.tomogram.global_cps(i=i, binsize=binsize).mean(axis="r")
                ]
                self._peaks = None
            self["pos"].visible = False
        if self.pos == 0 or self.show_what == "global-CFT":
            self._pos_changed(0)
        else:
            self.pos = 0
        self.canvas.auto_range()
        self._current_binsize = binsize

        if (img := self._power_spectra[0]) is not None:
            shape = img.shape
            center = np.ceil(np.array(shape) / 2 - 0.5)[::-1]
            self._infline_x.angle = 0
            self._infline_x.pos = center
            self._infline_y.angle = 90
            self._infline_y.pos = center
        self._upsampled_image_item.setVisible(False)

    def _update_image(self):
        if self._is_log_scale:
            self.canvas.image = np.log(self._image + 1e-12)
        else:
            self.canvas.image = self._image

    def _upsample_and_update_image(
        self,
        a: float,
        y: float,
        radius: float = 3.0,
        upsample: int = 10,
    ):
        from cylindra.components._peak import PeakDetector

        if self._peaks is None:
            return None
        img = self._peaks[self.pos].image
        det = PeakDetector(img, nsamples=2)
        if (
            y < radius
            or y > img.shape.y - radius
            or a < radius
            or a > img.shape.a - radius
        ):
            return None
        yloc = y - self._image.shape[0] // 2
        aloc = a - self._image.shape[1] // 2
        ps, yoff, aoff = det._local_ps_and_offset(
            (yloc - radius, yloc + radius),
            (aloc - radius, aloc + radius),
            up_y=upsample,
            up_a=upsample,
        )
        self._upsampled_image_item.setImage(ps.value.T)
        # move image layer to the correct position
        scale = 1 / upsample
        y0 = yoff * scale + self._image.shape[0] // 2
        a0 = aoff * scale + self._image.shape[1] // 2
        tr = self._upsampled_image_item.transform()
        tr.setMatrix(scale, 0, 0, 0, scale, 0, a0, y0, 1)
        self._upsampled_image_item.setTransform(tr)
        self._upsampled_image_item.setLevels(self.canvas.contrast_limits)
        self._upsampled_image_item.setVisible(True)
        self._last_upsample_params = {
            "a": a,
            "y": y,
            "radius": radius,
            "upsample": upsample,
        }

    @pos.connect
    def _pos_changed(self, pos: int):
        if len(self._power_spectra) == 0:
            return None
        _next_image = self._power_spectra[pos]
        if self._may_show_text_overlay(_next_image is None, "No CFT available"):
            return None
        self._image = np.asarray(_next_image)
        self._update_image()
        if self._peaks is None:
            self._markers.visible = False
        else:
            x = [peak.a for peak in self._peaks[pos].peaks]
            y = [peak.y for peak in self._peaks[pos].peaks]
            self._markers.data = (x, y)
            self._markers.visible = True
        if self._upsampled_image_item.isVisible() and self._last_upsample_params:
            self._upsample_and_update_image(**self._last_upsample_params)
        return None

    @show_what.connect
    def _show_what_changed(self, value: str):
        if len(self._power_spectra) > 0:
            self._set_spline(self._current_spline_index, binsize=None)

    def _may_show_text_overlay(self, when: bool, text: str) -> bool:
        if when:
            self.canvas.text_overlay.text = text
            self.canvas.text_overlay.visible = True
            self.canvas.text_overlay.color = "lime"
            del self.canvas.image
            self._markers.visible = False
            self._image = np.zeros((1, 1))
        else:
            self.canvas.text_overlay.visible = False
        return when


@magicclass(widget_type="groupbox", record=False)
class Parameters(MagicTemplate):
    """
    Cylinder paramters.

    Attributes
    ----------
    spacing : str
        Lattice spacing.
    rise : str
        Rise angle (degree).
    twist : str
        Skew angle (degree).
    npf : str
        Number of protofilaments.
    """

    radius = vfield("").with_options(enabled=False)
    spacing = vfield("").with_options(enabled=False)
    rise = vfield("").with_options(enabled=False)
    twist = vfield("").with_options(enabled=False)
    npf = vfield("").with_options(enabled=False)
    rise_sign = vfield("").with_options(enabled=False)

    def __init__(self):
        self._radius = None
        self._spacing = None
        self._rise = None
        self._twist = None
        self._npf = None
        self._rise_sign = None

    @set_design(text="Export as CSV ...")
    def export(self, path: Path.Save[FileFilter.CSV]):
        import polars as pl

        return pl.DataFrame(
            {
                "radius": [self.radius],
                "spacing": [self.spacing],
                "rise": [self.rise],
                "twist": [self.twist],
                "npf": [self.npf],
                "rise_sign": [self.rise_sign],
            }
        ).write_csv(path)

    @radius.post_get_hook
    def _get_radius(self, value):
        return self._radius

    @radius.pre_set_hook
    def _set_radius(self, value):
        self._radius = value
        if value is None:
            return "-- nm"
        else:
            return f"{value:.2f} nm"

    @spacing.post_get_hook
    def _get_spacing(self, value):
        return self._spacing

    @spacing.pre_set_hook
    def _set_spacing(self, value):
        self._spacing = value
        if value is None:
            return "-- nm"
        else:
            return f"{value:.2f} nm"

    @rise.post_get_hook
    def _get_rise(self, value):
        return self._rise

    @rise.pre_set_hook
    def _set_rise(self, value):
        self._rise = value
        if value is None:
            return "--°"
        else:
            return f"{value:.2f}°"

    @twist.post_get_hook
    def _get_twist(self, value):
        return self._twist

    @twist.pre_set_hook
    def _set_twist(self, value):
        self._twist = value
        if value is None:
            return "--°"
        else:
            return f"{value:.2f}°"

    @npf.post_get_hook
    def _get_npf(self, value):
        return self._npf

    @npf.pre_set_hook
    def _set_npf(self, value):
        self._npf = value
        if value is None:
            return "--"
        else:
            return f"{int(value)}"

    @rise_sign.post_get_hook
    def _get_rise_sign(self, value):
        return self._rise_sign

    @rise_sign.pre_set_hook
    def _set_rise_sign(self, value):
        self._rise_sign = value
        if self._rise_sign is None:
            return "--"
        else:
            return f"{int(value)}"


@magicclass(layout="horizontal", record=False)
class SpectraInspector(ChildWidget):
    """
    Widget to measure the periodicity of a tomographic structure.

    Attributes
    ----------
    parameters : Parameters
        Cylinder parameters measured manually.
    log_scale : bool
        Check to use log power spectra.
    """

    peak_viewer = field(PeakInspector)

    def __post_init__(self) -> None:
        self.mode = MouseMode.none
        self.SidePanel.native.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

    @magicclass(properties={"min_width": 200})
    class SidePanel(MagicTemplate):
        """
        Measure/inspect spectrum.

        Attributes
        ----------
        current_spline : str
            Current spline whose power spectrum is being displayed.
        current_bin_size : str
            Current bin size used for calculating the power spectrum (maybe different
            from the bin size of the tomogram).
        radius : float
            Radius to upsample locally.
        factor : int
            Upsampling factor used for the "Upsample spectrum" function.
        """

        parameters = abstractapi()
        load_spline = abstractapi()
        current_spline = vfield("").with_options(enabled=False)
        set_bin_size = abstractapi()
        current_bin_size = vfield("").with_options(enabled=False)
        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
        upsample_spectrum = abstractapi()
        radius = vfield(3.0).with_options(min=1, max=10, step=0.5)
        factor = vfield(10).with_options(min=2, max=50)
        log_scale = abstractapi()

    parameters = field(Parameters, location=SidePanel, name="Measured parameters")
    log_scale = vfield(False, location=SidePanel)

    @property
    def canvas(self):
        return self.peak_viewer.canvas

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        value = MouseMode(value)

        # update button texts
        btn_axial = get_button(self.select_axial_peak)
        btn_angular = get_button(self.select_angular_peak)
        btn_upsample = get_button(self.upsample_spectrum)
        if value is not MouseMode.none and self.mode is value:
            value = MouseMode.none
        match value:
            case MouseMode.none:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Select angular peak"
                btn_upsample.text = "Upsample spectrum"
                self.peak_viewer._upsampled_image_item.setVisible(False)
            case MouseMode.axial:
                btn_axial.text = "Selecting ..."
                btn_angular.text = "Select angular peak"
                btn_upsample.text = "Upsample spectrum"
                self.peak_viewer._upsampled_image_item.setVisible(False)
            case MouseMode.angular:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Selecting ..."
                btn_upsample.text = "Upsample spectrum"
                self.peak_viewer._upsampled_image_item.setVisible(False)
            case MouseMode.upsample:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Select angular peak"
                btn_upsample.text = "Upsampling ..."
            case _:  # pragma: no cover
                raise ValueError(f"Invalid mode: {value}")
        self._mode = value

    def _get_splines(self, _=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        tomo = self._get_main().tomogram
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_binsize_choices(self, *_) -> list[int]:
        parent = self._get_main()
        return [k for k, _ in parent.tomogram.multiscaled]

    @set_design(text="Load spline", location=SidePanel)
    def load_spline(
        self,
        idx: Annotated[int, {"choices": _get_splines}],
        binsize: Annotated[int, {"bind": None}] = None,
    ):
        """Load a spline from the spline list to this widget."""
        self.canvas.mouse_clicked.disconnect(self._on_mouse_clicked, missing_ok=True)
        parent = self._get_main()
        tomo = parent.tomogram
        spl = tomo.splines[idx]
        self.parameters.radius = spl.radius
        self.parameters.rise_sign = spl.config.rise_sign
        self.parameters.spacing = None
        self.parameters.rise = None
        self.parameters.twist = None
        self.parameters.npf = None

        self._on_log_scale_changed(self.log_scale)
        self.peak_viewer._set_spline(idx, binsize)

        self.canvas.mouse_clicked.connect(self._on_mouse_clicked, unique=True)
        self.mode = MouseMode.none
        self.SidePanel.current_bin_size = str(self.peak_viewer._current_binsize)
        self.SidePanel.current_spline = str(idx)

    @set_design(text=capitalize, location=SidePanel)
    def set_bin_size(self, binsize: Annotated[int, {"choices": _get_binsize_choices}]):
        """Override the current bin size."""
        self.load_spline(int(self.SidePanel.current_spline), binsize)

    @set_design(text=capitalize, location=SidePanel)
    def select_axial_peak(self):
        """Click to start selecting the axial peak for measurement."""
        self.mode = MouseMode.axial

    @set_design(text=capitalize, location=SidePanel)
    def select_angular_peak(self):
        """Click to start selecting the angular peak for measurement."""
        if self.parameters.spacing is None:
            raise ValueError("Select the axial peak first.")
        self.mode = MouseMode.angular

    @set_design(text=capitalize, location=SidePanel)
    def upsample_spectrum(self):
        """Click to turn on upsampling the power spectrum."""
        if self.mode is MouseMode.upsample:
            self.mode = MouseMode.none
        else:
            self.mode = MouseMode.upsample

    @log_scale.connect
    def _on_log_scale_changed(self, value: bool):
        self.peak_viewer._is_log_scale = value
        self.peak_viewer._update_image()

    def _on_mouse_clicked(self, e: mouse_event.MouseClickEvent):
        return self._click_at(e.pos())

    def _click_at(self, pos: tuple[float, float]):
        if self.mode == MouseMode.none:
            return
        a0, y0 = pos
        shape = self.canvas.image.shape
        ycenter, acenter = np.ceil(np.array(shape) / 2 - 0.5)
        afreq = (a0 - acenter) / shape[1]
        yfreq = (y0 - ycenter) / shape[0]

        parent = self._get_main()
        scale = parent.tomogram.scale

        match self.mode:
            case MouseMode.axial:
                self.parameters.spacing = abs(1.0 / yfreq * scale) * int(
                    self.SidePanel.current_bin_size
                )
                _sign = self.parameters.rise_sign
                self.parameters.rise = np.rad2deg(np.arctan(afreq / yfreq)) * _sign
                self.peak_viewer._layer_axial.data = [a0], [y0]
                self.mode = MouseMode.none
            case MouseMode.angular:
                _p = self.parameters
                self.parameters.twist = np.rad2deg(
                    np.arctan(yfreq / afreq * _p.spacing / _p.radius)
                )
                self.parameters.npf = int(round(abs(a0 - acenter)))
                self.peak_viewer._layer_angular.data = [a0], [y0]
                self.mode = MouseMode.none
            case MouseMode.upsample:
                self.peak_viewer._upsample_and_update_image(
                    a0, y0, self.SidePanel.radius, self.SidePanel.factor
                )
            case _:
                pass
