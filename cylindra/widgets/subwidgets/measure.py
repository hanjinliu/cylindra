from enum import Enum
from typing import TYPE_CHECKING, Annotated

import impy as ip
import numpy as np
import polars as pl
from magicclass import (
    abstractapi,
    field,
    get_button,
    magicclass,
    nogui,
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


LOCAL_CFT = "Local-CFT"
LOCAL_CFT_UP = "Upsampled local-CFT"
GLOBAL_CFT = "Global-CFT"
GLOBAL_CFT_UP = "Upsampled global-CFT"

UPSAMPLE = 5


@magicclass(record=False)
class PeakInspector(ChildWidget):
    show_what = vfield(
        GLOBAL_CFT, widget_type="RadioButtons", name="Show:"
    ).with_choices([LOCAL_CFT, LOCAL_CFT_UP, GLOBAL_CFT, GLOBAL_CFT_UP])
    canvas = field(QtImageCanvas)
    pos = vfield(int, widget_type="Slider", label="Position").with_options(max=0)

    def _set_peaks(self, peaks: "list[ImageWithPeak]", upsample: int = 1):
        if upsample == 1:
            self._power_spectra = [peak.power() for peak in peaks]
        else:
            self._power_spectra = [
                peak.power_upsampled(upsample=upsample) for peak in peaks
            ]
        self._peaks = peaks

    def reset_choices(self, *_):
        pass

    def __init__(self):
        self._peaks: list[ImageWithPeak] = []
        self._power_spectra = list[ip.ImgArray | None]()
        self._image = np.zeros((1, 1))
        self._is_log_scale = False

    def __post_init__(self):
        self._infline_x = self.canvas.add_infline((0, 0), 0, color="yellow")
        self._infline_y = self.canvas.add_infline((0, 0), 90, color="yellow")
        self._border_line = self.canvas.add_curve([], [], color="gray")
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
                self._peaks = []
                self._image = np.zeros((1, 1))
                return None
            else:
                raise ValueError(f"Invalid spline index: {i}")
        spl = main.splines[i]
        self._layer_axial.data = [], []
        self._layer_angular.data = [], []
        ins = self.find_ancestor(SpectraInspector)
        ins.SidePanel.current_spline.value = i
        upsample = 1 if self.show_what in (LOCAL_CFT, GLOBAL_CFT) else UPSAMPLE
        if self.show_what in (LOCAL_CFT, LOCAL_CFT_UP):
            if spl.props.has_loc(H.twist) and spl.has_anchors:
                # has local-CFT results
                if binsize is None:
                    binsize = spl.props.binsize_loc[H.twist]
                peaks = main.tomogram.local_image_with_peaks(i=i, binsize=binsize)
                self._set_peaks(peaks, upsample=upsample)
            else:
                # will not show anything
                self._power_spectra = [None]
                self._peaks = []
                binsize = 1
            self._update_slider_for_power_spectra(True, scale=1.0 / upsample)
        elif self.show_what in (GLOBAL_CFT, GLOBAL_CFT_UP):
            if self._may_show_text_overlay(
                spl.radius is None, "No CFT available (radius not set)"
            ):
                return
            if binsize is None:
                if spl.props.has_glob(H.twist):
                    binsize = spl.props.binsize_glob[H.twist]
                else:
                    main = self._get_main()
                    binsize = roundint(
                        main._reserved_layers.scale / main.tomogram.scale
                    )
            peak = [main.tomogram.global_image_with_peaks(i=i, binsize=binsize)]
            self._set_peaks(peak, upsample=upsample)
            self._update_slider_for_power_spectra(False, scale=1.0 / upsample)
        else:  # pragma: no cover
            raise RuntimeError(f"Unreachable: {self.show_what=}")

        ins.SidePanel.current_bin_size.value = binsize

        if (img := self._power_spectra[0]) is not None:
            shape = self._fix_shape(img.shape)
            center = np.ceil(np.array(shape) / 2 - 0.5)[::-1]
            self._infline_x.angle = 0
            self._infline_x.pos = center
            self._infline_y.angle = 90
            self._infline_y.pos = center
            border_data = self._calc_image_border_line()
            self._border_line.data = border_data
            self.canvas.xlim = border_data[0][0], border_data[0][2]
            self.canvas.ylim = border_data[1][0], border_data[1][2]

    def _update_slider_for_power_spectra(self, visible: bool, scale: float = 1.0):
        self["pos"].visible = visible
        if visible:
            self["pos"].max = len(self._power_spectra) - 1
        self.canvas.image_scale = scale
        if self.pos == 0 or self.show_what == GLOBAL_CFT:
            self._pos_changed(0)
        else:
            self.pos = 0

    def _calc_image_border_line(self):
        if isinstance(self.canvas.image_scale, tuple):
            yscale, xscale = self.canvas.image_scale
        else:
            yscale = xscale = self.canvas.image_scale
        shape = self._image.shape
        len_x = shape[1] * xscale
        len_y = shape[0] * yscale
        return (
            np.array([0, len_x, len_x, 0, 0]) - 0.5,
            np.array([0, 0, len_y, len_y, 0]) - 0.5,
        )

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
        if self._may_show_text_overlay(_next_image is None, "No CFT available"):
            return None
        self._image = np.asarray(_next_image)
        self._update_image()
        x = np.array([peak.a for peak in self._peaks[pos].peaks])
        y = np.array([peak.y for peak in self._peaks[pos].peaks])
        self._markers.data = (x, y)
        self._markers.visible = True

    @show_what.connect
    def _show_what_changed(self, value: str):
        if len(self._power_spectra) > 0:
            ins = self.find_ancestor(SpectraInspector)
            idx = ins.SidePanel.current_spline.value
            binsize = ins.SidePanel.current_bin_size.value
            self._set_spline(idx, binsize=binsize)

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

    def _fix_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        if self.show_what in (LOCAL_CFT_UP, GLOBAL_CFT_UP):
            shape = tuple(s // UPSAMPLE for s in shape)
        return shape


@magicclass(widget_type="groupbox", record=False)
class Parameters(ChildWidget):
    """Cylinder paramters.

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

    def _to_polars(self):
        return pl.DataFrame(
            {
                "name": ["radius", "spacing", "rise", "twist", "npf", "rise_sign"],
                "value": [
                    self.radius,
                    self.spacing,
                    self.rise,
                    self.twist,
                    self.npf,
                    self.rise_sign,
                ],
            }
        )

    @set_design(text="Log to console")
    def log_to_console(self):
        """Log these parameters to the console."""
        main = self._get_main()
        main.logger.print_table(self._to_polars())

    @set_design(text="Copy to clipboard")
    def copy_to_clipboard(self):
        """Copy these parameters to clipboard as a tab-separated string."""
        self._to_polars().write_clipboard()

    @set_design(text="Export as CSV ...")
    def export(self, path: Path.Save[FileFilter.CSV]):
        """Export these parameters as a CSV file."""
        self._to_polars().write_csv(path)

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
    """Widget to measure the periodicity of a tomographic structure.

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
    class SidePanel(ChildWidget):
        parameters = abstractapi()

        @magicclass(labels=False, layout="horizontal")
        class current_spline(ChildWidget):
            """Current spline whose power spectrum is being displayed."""

            label_text = vfield("Spline:", widget_type="Label")
            param_value = vfield("0").with_options(enabled=False)

            def _get_splines(self, _=None) -> list[tuple[str, int]]:
                """Get list of spline objects for categorical widgets."""
                tomo = self._get_main().tomogram
                return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

            @set_design(text="Set ...")
            def set_spline(self, idx: Annotated[int, {"choices": _get_splines}]):
                """Override the current bin size."""
                ins = self.find_ancestor(SpectraInspector)
                ins.load_spline(idx, ins.SidePanel.current_bin_size.value)

            @property
            def value(self):
                return int(self.param_value)

            @value.setter
            def value(self, v: int):
                self.param_value = str(v)

        @magicclass(labels=False, layout="horizontal")
        class current_bin_size(ChildWidget):
            """Current bin size used for calculating the power spectrum (maybe different
            from the bin size of the tomogram)."""

            label_text = vfield("Bin size:", widget_type="Label")
            param_value = vfield("1").with_options(enabled=False)

            def _get_binsize_choices(self, *_) -> list[int]:
                parent = self._get_main()
                binsizes = [k for k, _ in parent.tomogram.multiscaled]
                if 1 not in binsizes:
                    binsizes.append(1)
                return sorted(binsizes)

            @set_design(text="Set ...")
            def set_bin_size(
                self, binsize: Annotated[int, {"choices": _get_binsize_choices}]
            ):
                """Override the current bin size."""
                ins = self.find_ancestor(SpectraInspector)
                ins.load_spline(ins.SidePanel.current_spline.value, binsize)

            @property
            def value(self):
                return int(self.param_value)

            @value.setter
            def value(self, v: int):
                self.param_value = str(v)

        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
        log_scale = abstractapi()
        local_cft_for_this = abstractapi()

    parameters = field(Parameters, location=SidePanel, name="Measured parameters")
    log_scale = vfield(False, location=SidePanel)

    @set_design(text="Run local-CFT for this", location=SidePanel)
    def local_cft_for_this(
        self,
        interval: Annotated[float, {"min": 1.0, "step": 0.5}] = 50.0,
        depth: Annotated[float, {"min": 2.0, "step": 0.5}] = 50.0,
    ):
        """Run local-CFT for the current spline."""
        main = self._get_main()
        idx = self.SidePanel.current_spline.value
        if idx >= len(main.splines):
            raise ValueError("No spline available.")
        bin_size = self.SidePanel.current_bin_size.value
        main.local_cft_analysis(
            [idx],
            interval=interval,
            depth=depth,
            bin_size=bin_size,
        )
        self.load_spline(idx, binsize=bin_size)

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
        if value is not MouseMode.none and self.mode is value:
            value = MouseMode.none
        match value:
            case MouseMode.none:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Select angular peak"
            case MouseMode.axial:
                btn_axial.text = "Selecting ..."
                btn_angular.text = "Select angular peak"
            case MouseMode.angular:
                btn_axial.text = "Select axial peak"
                btn_angular.text = "Selecting ..."
            case _:  # pragma: no cover
                raise ValueError(f"Invalid mode: {value}")
        self._mode = value

    @nogui
    def load_spline(
        self,
        idx: int,
        binsize: int | None = None,
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
        self.SidePanel.current_spline.value = idx

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

    @log_scale.connect
    def _on_log_scale_changed(self, value: bool):
        self.peak_viewer._is_log_scale = value
        self.peak_viewer._update_image()

    def _on_mouse_clicked(self, e: mouse_event.MouseClickEvent):
        return self._click_at(e.pos())

    def _click_at(self, pos: tuple[float, float]):
        if self.mode is MouseMode.none:
            return
        a0, y0 = pos
        shape = self.peak_viewer._fix_shape(self.canvas.image.shape)
        ycenter, acenter = np.ceil(np.array(shape) / 2 - 0.5)
        afreq = (a0 - acenter) / shape[1]
        yfreq = (y0 - ycenter) / shape[0]

        parent = self._get_main()
        scale = parent.tomogram.scale

        match self.mode:
            case MouseMode.axial:
                self.parameters.spacing = (
                    abs(1.0 / yfreq * scale) * self.SidePanel.current_bin_size.value
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
