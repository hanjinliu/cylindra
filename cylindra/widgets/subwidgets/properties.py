from typing import TYPE_CHECKING, Annotated

from magicclass import (
    FieldGroup,
    MagicTemplate,
    abstractapi,
    box,
    field,
    logging,
    magicclass,
    set_design,
    vfield,
)
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from magicclass.types import Path
from psygnal import Signal

from cylindra.const import FileFilter, Ori
from cylindra.const import PropertyNames as H
from cylindra.widget_utils import capitalize
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components import CylSpline

_Logger = logging.getLogger("cylindra")


class LabeledText(FieldGroup):
    lbl = field("", widget_type="Label")
    txt = vfield("").with_options(enabled=False)

    def __init__(self, label_text: str):
        super().__init__(labels=False)
        self.lbl.value = label_text
        self.margins = (0, 0, 0, 0)


_PLOT_DEFAULTS = {
    H.spacing: ("spacing (nm)", "lime"),
    H.pitch: ("pitch (nm)", "green"),
    H.twist: ("twist (deg)", "gold"),
    H.skew: ("skew angle (deg)", "orange"),
    H.rise: ("rise angle (deg)", "cyan"),
    H.rise_length: ("rise length (nm)", "lightblue"),
    H.radius: ("radius (nm)", "magenta"),
    H.npf: ("PF number", "lightgray"),
    H.start: ("start", "violet"),
}


@magicclass(name="Local Properties", record=False)
class LocalPropertiesWidget(ChildWidget):
    """
    Local properties.

    Attributes
    ----------
    plot : QtMultiPlotCanvas
        Plot of local properties
    """

    _props_changed = Signal(list[str])

    @magicclass(
        widget_type="groupbox",
        layout="horizontal",
        labels=False,
        name="lattice parameters",
    )
    class params(MagicTemplate):
        """Structural parameters at the current position"""

        spacing = LabeledText("spacing")
        twist = LabeledText("twist")
        rise = LabeledText("rise angle")
        structure = LabeledText("structure")

    plot = box.resizable(
        field(QtMultiPlotCanvas).with_options(
            nrows=3, ncols=1, sharex=True, tooltip="Plot of local properties"
        ),
        x_enabled=False,
    )

    def __post_init__(self):
        # Initialize multi-plot canvas
        self.plot.min_height = 300
        self._props_to_plot = list[str]()
        self._set_properties_to_plot([H.spacing, H.twist, H.rise])
        self._init_text()

    def _init_text(self):
        self.params.spacing.txt = " -- nm"
        self.params.twist.txt = " -- °"
        self.params.rise.txt = " -- °"
        self.params.structure.txt = " -- "
        return None

    def _set_text(self, spl: "CylSpline", i: int):
        self.params.spacing.txt = f" {spl.props.get_loc(H.spacing)[i]:.2f} nm"
        self.params.twist.txt = f" {spl.props.get_loc(H.twist)[i]:.2f}°"
        self.params.rise.txt = f" {spl.props.get_loc(H.rise)[i]:.2f}°"
        npf = int(spl.props.get_loc(H.npf)[i])
        start = spl.props.get_loc(H.start)[i]
        self.params.structure.txt = f" {npf}_{start}"
        return None

    def _init_plot(self):
        for _plot in self.plot:
            _plot.layers.clear()
        return None

    def _set_properties_to_plot(self, props: list[str]):
        # first insert/remove axes to match the number of properties
        nplots = len(self.plot)
        for i in range(nplots, len(props)):
            self.plot.addaxis(i + 1, 0)
        for _ in range(len(props), nplots):
            del self.plot[0]
        # update all the lines
        for prop, _plot in zip(props, self.plot, strict=True):
            info = _PLOT_DEFAULTS.get(prop, ("unknown", "lightgray"))
            _plot.ylabel = info[0]
            _plot.legend.visible = False
            _plot.border = [1, 1, 1, 0.2]
        if len(self.plot) > 0:
            self.plot[-1].xlabel = "position (nm)"
        self._props_to_plot = props.copy()

    def _plot_properties(self, spl: "CylSpline"):
        if not spl.has_anchors:
            return None
        x = spl.anchors * spl.length()
        if x[0] > x[-1]:
            x = x[::-1]

        self._init_plot()

        kw = {"pos": [x[0], 0], "degree": 90, "color": [1.0, 0.0, 0.0, 0.3], "lw": 2}
        for prop, _plot in zip(self._props_to_plot, self.plot, strict=True):
            if (_interv := spl.props.get_loc(prop, None)) is not None:
                color = _PLOT_DEFAULTS[prop][1]
                _plot.add_curve(x, _interv, color=color, antialias=True)
            _plot.add_infline(**kw)
        if len(self.plot) > 0:
            self.plot[0].xlim = (x[0] - 2, x[-1] + 2)
        self._plot_spline_position(x[0])
        return None

    def _plot_spline_position(self, x: float):
        """update current position indicator (the red vertical line)"""
        for _plot in self.plot:
            if len(_plot.layers) > 0:
                _plot.layers[-1].pos = [x, 0]
        if len(self.plot) > 0:
            first_plot = self.plot[0]
            xmin, xmax = first_plot.xlim
            if len(self.plot) and (x < xmin or xmax < x):
                dx = xmax - xmin
                first_plot.xlim = (x - dx / 2, x + dx / 2)
        return None

    @magicclass(layout="horizontal", labels=False, record=False)
    class footer(MagicTemplate):
        edit_plots = abstractapi()
        copy_screenshot = abstractapi()
        save_screenshot = abstractapi()
        log_screenshot = abstractapi()

    @set_design(text=capitalize, location=footer)
    def edit_plots(
        self,
        props: Annotated[list[str], {"widget_type": CheckBoxes, "choices": _PLOT_DEFAULTS.keys()}] = (H.spacing, H.twist, H.rise)
    ):  # fmt: skip
        """Edit which properties to plot."""
        self._set_properties_to_plot(props)
        self._props_changed.emit(props)
        return None

    @set_design(max_width=40, text="Copy", location=footer)
    def copy_screenshot(self):
        """Copy the screenshot of the plots to clipboard."""
        return self.plot.to_clipboard()

    @set_design(max_width=40, text="Scr", location=footer)
    def save_screenshot(self, path: Path.Save[FileFilter.PNG]):
        """Take a screenshot of the plots."""
        from skimage.io import imsave

        img = self.plot.render()
        return imsave(path, img)

    @set_design(max_width=40, text="Log", location=footer)
    def log_screenshot(self):
        """Take a screenshot of the plots and show in the logger."""
        import matplotlib.pyplot as plt

        img = self.plot.render()
        with _Logger.set_plt():
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return None


@magicclass(widget_type="collapsible", name="Global Properties")
class GlobalPropertiesWidget(MagicTemplate):
    """Global properties."""

    def __post_init__(self):
        self._init_text()

    @magicclass(
        widget_type="groupbox",
        labels=False,
        name="lattice parameters",
        properties={"margins": (0, 0, 0, 0)},
    )
    class params(MagicTemplate):
        @magicclass(
            layout="horizontal", labels=False, properties={"margins": (0, 0, 0, 0)}
        )
        class params1(MagicTemplate):
            spacing = LabeledText("spacing")
            twist = LabeledText("twist")
            rise = LabeledText("rise angle")
            structure = LabeledText("structure")

        @magicclass(
            layout="horizontal", labels=False, properties={"margins": (0, 0, 0, 0)}
        )
        class params2(MagicTemplate):
            radius = LabeledText("radius")
            orientation_ = LabeledText("orientation")

    def _init_text(self):
        self.params.params1.spacing.txt = " -- nm"
        self.params.params1.twist.txt = " -- °"
        self.params.params1.rise.txt = " -- °"
        self.params.params1.structure.txt = " -- "
        self.params.params2.radius.txt = " -- nm"
        self.params.params2.orientation_.txt = " -- "
        return None

    def _set_text(self, spl: "CylSpline"):
        self.params.params1.spacing.txt = f" {_fmt_prop(spl, H.spacing)} nm"
        self.params.params1.twist.txt = f" {_fmt_prop(spl, H.twist)}°"
        self.params.params1.rise.txt = f" {_fmt_prop(spl, H.rise)}°"
        npf = spl.props.get_glob(H.npf, None)
        start = spl.props.get_glob(H.start, None)
        if npf is None or start is None:
            self.params.params1.structure.txt = " -- "
        else:
            self.params.params1.structure.txt = f" {npf}_{start}"
        self.params.params2.radius.txt = f" {_fmt_prop(spl, H.radius)} nm"
        if spl.orientation is Ori.none:
            self.params.params2.orientation_.txt = " -- "
        else:
            self.params.params2.orientation_.txt = spl.orientation
        return None


def _fmt_prop(spl: "CylSpline", name: str) -> str:
    value = spl.props.get_glob(name, None)
    if value is None:
        return " -- "
    return f"{value:.2f}"
