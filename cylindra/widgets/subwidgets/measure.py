from enum import Enum
from typing import Annotated

import numpy as np
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
from cylindra.utils import roundint
from cylindra.widgets.subwidgets._child_widget import ChildWidget


class MeasureMode(Enum):
    none = "none"
    axial = "spacing/rise"
    angular = "twist/npf"


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
        Cylinder parameters.
    log_scale : bool
        Check to use log power spectra.
    """

    canvas = field(QtImageCanvas)

    def __post_init__(self) -> None:
        self._layer_axial = None
        self._layer_angular = None
        self._image = None
        self.mode = MeasureMode.none

    @magicclass(properties={"min_width": 200})
    class SidePanel(MagicTemplate):
        parameters = abstractapi()
        load_spline = abstractapi()
        set_binsize = abstractapi()
        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
        log_scale = abstractapi()

    parameters = field(Parameters, location=SidePanel)
    log_scale = vfield(False, location=SidePanel)

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

    def _get_binsize(self, *_) -> int:
        parent = self._get_main()
        return roundint(parent._reserved_layers.scale / parent.tomogram.scale)

    def _get_binsize_choices(self, *_) -> list[int]:
        parent = self._get_main()
        return [k for k, _ in parent.tomogram.multiscaled]

    @set_design(text="Load spline", location=SidePanel)
    def load_spline(
        self,
        idx: Annotated[int, {"bind": _get_current_index}],
        binsize: Annotated[int, {"bind": _get_binsize}] = 1,
    ):
        """Load current spline to the canvas."""
        self.canvas.mouse_clicked.disconnect(self._on_mouse_clicked, missing_ok=True)
        parent = self._get_main()
        tomo = parent.tomogram
        spl = tomo.splines[idx]
        self.parameters.radius = spl.radius
        self.parameters.rise_sign = spl.config.rise_sign
        polar = tomo.straighten_cylindric(idx, binsize=binsize)
        pw = polar.power_spectra(zero_norm=True, dims="rya").mean(axis="r")

        self.canvas.layers.clear()
        self._image = pw.value
        self.canvas.image = self._image
        self._on_log_scale_changed(self.log_scale)

        center = np.ceil(np.array(pw.shape) / 2 - 0.5)
        self.canvas.add_infline(center[::-1], 0, color="yellow")
        self.canvas.add_infline(center[::-1], 90, color="yellow")

        self.canvas.mouse_clicked.connect(self._on_mouse_clicked, unique=True)

    @set_design(text="Set bin size", location=SidePanel)
    def set_binsize(self, binsize: Annotated[int, {"choices": _get_binsize_choices}]):
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
        if value:
            self.canvas.image = np.log(self._image + 1e-12)
        else:
            self.canvas.image = self._image

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
            self.parameters.spacing = abs(1.0 / yfreq * scale) * self._get_binsize()
            _sign = self.parameters.rise_sign
            self.parameters.rise = np.rad2deg(np.arctan(afreq / yfreq)) * _sign

            if self._layer_axial in self.canvas.layers:
                self.canvas.layers.remove(self._layer_axial)
            self._layer_axial = self.canvas.add_scatter(
                [a0], [y0], color="cyan", symbol="+", size=12
            )

        elif self.mode == MeasureMode.angular:
            _p = self.parameters
            self.parameters.twist = np.rad2deg(
                np.arctan(yfreq / afreq * _p.spacing / _p.radius)
            )
            self.parameters.npf = int(round(abs(a0 - acenter)))

            if self._layer_angular in self.canvas.layers:
                self.canvas.layers.remove(self._layer_angular)
            self._layer_angular = self.canvas.add_scatter(
                [a0], [y0], color="lime", symbol="+", size=12
            )
        self.mode = MeasureMode.none
