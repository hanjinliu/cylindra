from enum import Enum
from typing import TYPE_CHECKING, Annotated

import numpy as np
import impy as ip
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

from cylindra.const import FileFilter
from cylindra.const import PropertyNames as H
from cylindra.utils import roundint
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components.tomogram._cyl_tomo import PowerSpectrumWithPeak


class MeasureMode(Enum):
    none = "none"
    axial = "spacing/rise"
    angular = "twist/npf"


@magicclass(labels=False)
class PeakInspector(ChildWidget):
    show_what = vfield("Global-CFT", widget_type="RadioButtons").with_choices(
        ["Local-CFT", "Global-CFT"]
    )
    canvas = field(QtImageCanvas)
    pos = vfield(int, widget_type="Slider", label="Position").with_options(max=0)

    def _set_peaks(self, peaks: "list[PowerSpectrumWithPeak]"):
        self._power_spectra = [peak.power for peak in peaks]
        self._peaks = peaks

    def __init__(self):
        self._peaks: "list[PowerSpectrumWithPeak] | None" = None
        self._power_spectra = list[ip.ImgArray | None]()
        self._image = ip.zeros((1, 1))
        self._is_log_scale = False
        self._current_binsize = 1

    def __post_init__(self):
        self._infline_x = self.canvas.add_infline(1, 0, color="yellow")
        self._infline_y = self.canvas.add_infline(1, 90, color="yellow")
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
                self._image = ip.zeros((1, 1))
            return None
        spl = main.splines[i]
        self._layer_axial.data = [], []
        self._layer_angular.data = [], []
        if self.show_what == "Local-CFT":
            if spl.props.has_loc(H.twist) and spl.has_anchors:
                # has local-CFT results
                if binsize is None:
                    binsize = spl.props.binsize_loc[H.twist]
                peaks = main.tomogram.local_cps_with_peaks(i=i, binsize=binsize)
                self._set_peaks(peaks)
            else:
                # will not show anything
                self._power_spectra = [None]
                self._peaks = None
                binsize = 1
            self["pos"].visible = True
            self["pos"].max = len(self._power_spectra) - 1
        else:
            if spl.props.has_glob(H.twist):
                # has global-CFT results
                peak = [main.tomogram.global_cps_with_peaks(i=i, binsize=binsize)]
                self._set_peaks(peak)
            else:
                # calculate global-CFT here (for manual measurement)
                if binsize is None:
                    if spl.props.has_glob(H.twist):
                        binsize = spl.props.binsize_glob[H.twist]
                    else:
                        main = self._get_main()
                        binsize = roundint(main._reserved_layers.scale / main.tomogram.scale)
                self._power_spectra = list(
                    [main.tomogram.global_cps(i=i, binsize=binsize).mean(axis="r")]
                )
                self._peaks = None
            self["pos"].visible = False
        self._pos_changed(0)
        self.canvas.auto_range()
        self._current_binsize = binsize

        shape = self._power_spectra[0].shape
        center = np.ceil(np.array(shape) / 2 - 0.5)[::-1]
        self._infline_x.angle = 0
        self._infline_x.pos = center
        self._infline_y.angle = 90
        self._infline_y.pos = center

    def _update_image(self):
        if self._is_log_scale:
            self.canvas.image = np.log(self._image + 1e-12)
        else:
            self.canvas.image = self._image
    
    @pos.connect
    def _pos_changed(self, pos: int):
        if len(self._power_spectra) == 0:
            return None
        _next_image = self._power_spectra[pos]
        if _next_image is None:
            self.canvas.text_overlay.text = "No CFT available"
            self.canvas.text_overlay.visible = True
            self.canvas.text_overlay.color = "lime"
            return
        self.canvas.text_overlay.visible = False
        self._image = _next_image
        self._update_image()
        if self._peaks is None:
            self._markers.visible = False
        else:
            x = [peak.a for peak in self._peaks[pos].peaks]
            y = [peak.y for peak in self._peaks[pos].peaks]
            self._markers.data = (x, y)
            self._markers.visible = True
        return None

    @show_what.connect
    def _show_what_changed(self, value: str):
        if len(self._power_spectra) > 0:
            self._set_spline(self.pos, binsize=None)


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
        self.mode = MeasureMode.none

    @magicclass(properties={"min_width": 200})
    class SidePanel(MagicTemplate):
        parameters = abstractapi()
        load_spline = abstractapi()
        set_binsize = abstractapi()
        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
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
        value = MeasureMode(value)

        # update button texts
        btn_axial = get_button(self.select_axial_peak)
        btn_angular = get_button(self.select_angular_peak)
        match value:
            case MeasureMode.none:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Select angular peak"
            case MeasureMode.axial:
                btn_axial.text = "Selecting ..."
                btn_angular.text = "Select angular peak"
            case MeasureMode.angular:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Selecting ..."
            case _:  # pragma: no cover
                raise ValueError(f"Invalid mode: {value}")
        self._mode = value

    def _get_current_index(self, *_) -> int:
        parent = self._get_main()
        return parent.SplineControl.num

    def _get_binsize_choices(self, *_) -> list[int]:
        parent = self._get_main()
        return [k for k, _ in parent.tomogram.multiscaled]

    @set_design(text="Load spline", location=SidePanel)
    def load_spline(
        self,
        idx: Annotated[int, {"bind": _get_current_index}],
        binsize: Annotated[int, {"bind": None}] = None,
    ):
        """Load current spline to the canvas."""
        self.canvas.mouse_clicked.disconnect(self._on_mouse_clicked, missing_ok=True)
        parent = self._get_main()
        tomo = parent.tomogram
        spl = tomo.splines[idx]
        self.parameters.radius = spl.radius
        self.parameters.rise_sign = spl.config.rise_sign

        self._on_log_scale_changed(self.log_scale)
        self.peak_viewer._set_spline(idx, binsize)
        self.canvas.mouse_clicked.connect(self._on_mouse_clicked, unique=True)
        self.mode = MeasureMode.none

    @set_design(text="Set bin size", location=SidePanel)
    def set_binsize(self, binsize: Annotated[int, {"choices": _get_binsize_choices}]):
        """Override the current bin size."""
        self.load_spline(self._get_current_index(), binsize)

    @set_design(text="Set axial peak", location=SidePanel)
    def select_axial_peak(self):
        """Click to start selecting the axial peak."""
        self.mode = MeasureMode.axial

    @set_design(text="Set angular peak", location=SidePanel)
    def select_angular_peak(self):
        """Click to start selecting the angular peak."""
        if self.parameters.spacing is None:
            raise ValueError("Select the axial peak first.")
        self.mode = MeasureMode.angular

    @log_scale.connect
    def _on_log_scale_changed(self, value: bool):
        self.peak_viewer._is_log_scale = value
        self.peak_viewer._update_image()

    def _on_mouse_clicked(self, e: mouse_event.MouseClickEvent):
        return self._click_at(e.pos())

    def _click_at(self, pos: tuple[float, float]):
        if self.mode == MeasureMode.none:
            return
        a0, y0 = pos
        shape = self.canvas.image.shape
        ycenter, acenter = np.ceil(np.array(shape) / 2 - 0.5)
        afreq = (a0 - acenter) / shape[1]
        yfreq = (y0 - ycenter) / shape[0]

        parent = self._get_main()
        scale = parent.tomogram.scale

        if self.mode == MeasureMode.axial:
            self.parameters.spacing = abs(1.0 / yfreq * scale) * self.peak_viewer._current_binsize
            _sign = self.parameters.rise_sign
            self.parameters.rise = np.rad2deg(np.arctan(afreq / yfreq)) * _sign
            self.peak_viewer._layer_axial.data = [a0], [y0]
            self.mode = MeasureMode.none

        elif self.mode == MeasureMode.angular:
            _p = self.parameters
            self.parameters.twist = np.rad2deg(
                np.arctan(yfreq / afreq * _p.spacing / _p.radius)
            )
            self.parameters.npf = int(round(abs(a0 - acenter)))
            self.peak_viewer._layer_angular.data = [a0], [y0]
            self.mode = MeasureMode.none
        else:
            self.mode = MeasureMode.none