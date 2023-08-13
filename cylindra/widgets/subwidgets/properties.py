from typing import TYPE_CHECKING, Annotated
from magicclass import (
    abstractapi,
    magicclass,
    field,
    set_design,
    vfield,
    MagicTemplate,
    FieldGroup,
    box,
    logging,
)
from magicclass.types import Path
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from cylindra.const import PropertyNames as H
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.widgets.widget_utils import FileFilter

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


_PlotInfo = {
    H.spacing: ("spacing (nm)", "lime"),
    H.skew: ("skew angle (deg)", "gold"),
    H.skew_tilt: ("skew tilt angle (deg)", "orange"),
    H.rise: ("rise angle (deg)", "cyan"),
    H.rise_length: ("rise length (nm)", "lightblue"),
    H.radius: ("radius (nm)", "magenta"),
    H.npf: ("PF number", "lightgray"),
    H.start: ("start", "violet"),
}


@magicclass(widget_type="collapsible", name="Local Properties", record=False)
class LocalPropertiesWidget(MagicTemplate):
    """
    Local properties.

    Attributes
    ----------
    plot : QtMultiPlotCanvas
        Plot of local properties
    """

    @magicclass(
        widget_type="groupbox",
        layout="horizontal",
        labels=False,
        name="lattice parameters",
    )
    class params(MagicTemplate):
        """Structural parameters at the current position"""

        spacing = LabeledText("spacing")
        skew = LabeledText("skew angle")
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
        self._set_properties_to_plot([H.spacing, H.skew, H.rise])
        self._init_text()

    def _init_text(self):
        self.params.spacing.txt = " -- nm"
        self.params.skew.txt = " -- °"
        self.params.rise.txt = " -- °"
        self.params.structure.txt = " -- "
        return None

    def _set_text(self, spl: "CylSpline", i: int):
        self.params.spacing.txt = f" {spl.props.get_loc(H.spacing)[i]:.2f} nm"
        self.params.skew.txt = f" {spl.props.get_loc(H.skew)[i]:.2f}°"
        self.params.rise.txt = f" {spl.props.get_loc(H.rise)[i]:.2f}°"
        npf = int(spl.props.get_loc(H.npf)[i])
        start = spl.props.get_loc(H.start)[i]
        self.params.structure.txt = f" {npf}_{start:.1f}"
        return None

    def _init_plot(self):
        for _plot in self.plot:
            _plot.layers.clear()
        return None

    def _set_properties_to_plot(self, props: list[str]):
        nplots = len(self.plot)
        for i in range(nplots, len(props)):
            self.plot.addaxis(i, 0)
        for i in range(len(props), nplots):
            del self.plot[0]
        for prop, _plot in zip(props, self.plot):
            info = _PlotInfo.get(prop, ("unknown", "lightgray"))
            _plot.ylabel = info[0]
            _plot.legend.visible = False
            _plot.border = [1, 1, 1, 0.2]
        if len(self.plot) > 0:
            self.plot[-1].xlabel = "position (nm)"
        self._props_to_plot = props.copy()

    def _plot_properties(self, spl: "CylSpline"):
        if (x := spl.props.get_loc(H.spl_dist, None)) is None:
            return None
        if x[0] > x[-1]:
            x = x[::-1]

        self._init_plot()

        kw = dict(pos=[x[0], 0], degree=90, color=[1.0, 0.0, 0.0, 0.3], lw=2)
        for prop, _plot in zip(self._props_to_plot, self.plot):
            if (_interv := spl.props.get_loc(prop, None)) is not None:
                _plot.add_curve(x, _interv, color=_PlotInfo[prop][1])
            _plot.add_infline(**kw)
        if len(self.plot) > 0:
            self.plot[0].xlim = (x[0] - 2, x[-1] + 2)
        self._plot_spline_position(x[0])
        return None

    def _plot_spline_position(self, x: float):
        # update current position indicator
        for _plot in self.plot:
            if len(_plot.layers) > 0:
                _plot.layers[-1].pos = [x, 0]
        xmin, xmax = self.plot[0].xlim
        if len(self.plot) and (x < xmin or xmax < x):
            dx = xmax - xmin
            self.plot[0].xlim = (x - dx / 2, x + dx / 2)
        return None

    @magicclass(layout="horizontal", labels=False, record=False)
    class footer(MagicTemplate):
        edit_props = abstractapi()
        copy_screenshot = abstractapi()
        save_screenshot = abstractapi()
        log_screenshot = abstractapi()

    @footer.wraps
    @set_design(text="Edit plots")
    def edit_props(
        self,
        props: Annotated[list[str], {"widget_type": CheckBoxes, "choices": _PlotInfo.keys()}] = (H.spacing, H.skew, H.rise)
    ):  # fmt: skip
        from cylindra.widgets.main import CylindraMainWidget

        self._set_properties_to_plot(props)
        main = self.find_ancestor(CylindraMainWidget)
        spl = main.tomogram.splines[main.SplineControl.num]
        self._plot_properties(spl)
        return None

    @footer.wraps
    @set_design(max_width=40, text="Copy")
    def copy_screenshot(self):
        """Copy a screenshot of the plots to clipboard."""
        return self.plot.to_clipboard()

    @footer.wraps
    @set_design(max_width=40, text="Scr")
    def save_screenshot(self, path: Path.Save[FileFilter.PNG]):
        """Take a screenshot of the plots."""
        from skimage.io import imsave

        img = self.plot.render()
        return imsave(path, img)

    @footer.wraps
    @set_design(max_width=40, text="Log")
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
            skew = LabeledText("skew angle")
            rise = LabeledText("rise angle")
            structure = LabeledText("structure")

        @magicclass(
            layout="horizontal", labels=False, properties={"margins": (0, 0, 0, 0)}
        )
        class params2(MagicTemplate):
            radius = LabeledText("radius")
            polarity = LabeledText("polarity")

    def _init_text(self):
        self.params.params1.spacing.txt = " -- nm"
        self.params.params1.skew.txt = " -- °"
        self.params.params1.rise.txt = " -- °"
        self.params.params1.structure.txt = " -- "
        self.params.params2.radius.txt = " -- nm"
        self.params.params2.polarity.txt = " -- "
        return None

    def _set_text(self, spl: "CylSpline"):
        self.params.params1.spacing.txt = f" {spl.props.get_glob(H.spacing):.2f} nm"
        self.params.params1.skew.txt = f" {spl.props.get_glob(H.skew):.2f}°"
        self.params.params1.rise.txt = f" {spl.props.get_glob(H.rise):.2f}°"
        npf = int(spl.props.get_glob(H.npf))
        start = spl.props.get_glob(H.start)
        self.params.params1.structure.txt = f" {npf}_{start:.1f}"
        if spl.radius is not None:
            self.params.params2.radius.txt = f" {spl.radius:.2f} nm"
        else:
            self.params.params2.radius.txt = " -- nm"
        self.params.params2.polarity.txt = spl.orientation
        return None
