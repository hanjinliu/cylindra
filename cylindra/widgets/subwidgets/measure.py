from typing import TYPE_CHECKING, Annotated
from enum import Enum

from magicclass import (
    magicclass,
    abstractapi,
    set_design,
    vfield,
    field,
    MagicTemplate,
    get_button,
)
from magicclass.types import Path, OneOf
from magicclass.ext.pyqtgraph import QtImageCanvas, mouse_event

import numpy as np

from cylindra.utils import roundint
from cylindra.widgets.widget_utils import FileFilter
from ._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components import CylSpline


class MeasureMode(Enum):
    none = "none"
    axial = "spacing/rise"
    angular = "skew/npf"


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
    skew : str
        Skew angle (degree).
    npf : str
        Number of protofilaments.
    """

    radius = vfield("").with_options(enabled=False)
    spacing = vfield("").with_options(enabled=False)
    rise = vfield("").with_options(enabled=False)
    skew = vfield("").with_options(enabled=False)
    npf = vfield("").with_options(enabled=False)

    def __init__(self):
        self._radius = None
        self._spacing = None
        self._rise = None
        self._skew = None
        self._npf = None

    @set_design(text="Export as CSV ...")
    def export(self, path: Path.Save[FileFilter.CSV]):
        import polars as pl

        return pl.DataFrame(
            {
                "radius": [self.radius],
                "spacing": [self.spacing],
                "rise": [self.rise],
                "skew": [self.skew],
                "npf": [self.npf],
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

    @skew.post_get_hook
    def _get_skew(self, value):
        return self._skew

    @skew.pre_set_hook
    def _set_skew(self, value):
        self._skew = value
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
        self._spline: "CylSpline | None" = None
        self.mode = MeasureMode.none

    @magicclass(properties={"min_width": 200})
    class SidePanel(MagicTemplate):
        parameters = abstractapi()
        load_spline = abstractapi()
        set_binsize = abstractapi()
        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
        log_scale = abstractapi()

    parameters = SidePanel.field(Parameters)
    log_scale = SidePanel.vfield(False)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        value = MeasureMode(value)

        # update button texts
        btn_axial = get_button(self.select_axial_peak)
        btn_angular = get_button(self.select_angular_peak)
        if value is MeasureMode.none:
            btn_axial.text = "Select axial peak"
            btn_angular.text = "Select angular peak"
        elif value is MeasureMode.axial:
            btn_axial.text = "Selecting ..."
            btn_angular.text = "Select angular peak"
        else:
            btn_axial.text = "Select axial peak"
            btn_angular.text = "Select ..."
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

    @SidePanel.wraps
    @set_design(text="Load spline")
    def load_spline(
        self,
        idx: Annotated[int, {"bind": _get_current_index}],
        binsize: Annotated[int, {"bind": _get_binsize}] = 1,
    ):
        """Load current spline to the canvas."""
        self.canvas.mouse_clicked.disconnect(self._on_mouse_clicked, missing_ok=True)
        parent = self._get_main()
        tomo = parent.tomogram
        self._spline = tomo.splines[idx]
        self.parameters.radius = self._spline.radius
        polar = tomo.straighten_cylindric(idx, binsize=binsize)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")

        self.canvas.layers.clear()
        self._image = pw.value
        self.canvas.image = self._image
        self._on_log_scale_changed(self.log_scale)

        center = np.ceil(np.array(pw.shape) / 2 - 0.5)
        self.canvas.add_infline(center[::-1], 0, color="yellow")
        self.canvas.add_infline(center[::-1], 90, color="yellow")

        self.canvas.mouse_clicked.connect(self._on_mouse_clicked, unique=True)

    @SidePanel.wraps
    @set_design(text="Set bin size")
    def set_binsize(self, binsize: OneOf[_get_binsize_choices]):
        self.load_spline(self._get_current_index(), binsize)

    @SidePanel.wraps
    def select_axial_peak(self):
        """Click to start selecting the axial peak."""
        self.mode = MeasureMode.axial

    @SidePanel.wraps
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
        if self.mode == MeasureMode.none:
            return
        a0, y0 = e.pos()
        shape = self.canvas.image.shape
        ycenter, acenter = np.ceil(np.array(shape) / 2 - 0.5)
        afreq = (a0 - acenter) / shape[1]
        yfreq = (y0 - ycenter) / shape[0]

        parent = self._get_main()
        scale = parent.tomogram.scale

        if self.mode == MeasureMode.axial:
            self.parameters.spacing = abs(1.0 / yfreq * scale) * self._get_binsize()
            _sign = self._spline.config.rise_sign
            self.parameters.rise = np.rad2deg(np.arctan(afreq / yfreq)) * _sign

            if self._layer_axial in self.canvas.layers:
                self.canvas.layers.remove(self._layer_axial)
            self._layer_axial = self.canvas.add_scatter(
                [a0], [y0], color="cyan", symbol="+", size=12
            )

        elif self.mode == MeasureMode.angular:
            _p = self.parameters
            self.parameters.skew = np.rad2deg(
                np.arctan(yfreq / afreq * 2 * _p.spacing / _p.radius)
            )
            self.parameters.npf = int(round(abs(a0 - acenter)))

            if self._layer_angular in self.canvas.layers:
                self.canvas.layers.remove(self._layer_angular)
            self._layer_angular = self.canvas.add_scatter(
                [a0], [y0], color="lime", symbol="+", size=12
            )
        self.mode = MeasureMode.none
