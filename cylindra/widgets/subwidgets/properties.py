from typing import TYPE_CHECKING
from magicclass import (
    magicclass,
    field,
    vfield,
    MagicTemplate,
    FieldGroup,
)
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from cylindra.components import CylSpline


class LabeledText(FieldGroup):
    lbl = field("pitch", widget_type="Label")
    txt = vfield("").with_options(enabled=False)

    def __init__(self, label_text: str):
        super().__init__(labels=False)
        self.lbl.value = label_text
        self.margins = (0, 0, 0, 0)


@magicclass(widget_type="collapsible", name="Local Properties")
class LocalPropertiesWidget(MagicTemplate):
    """Local properties."""

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

    plot = field(QtMultiPlotCanvas, name="Plot").with_options(
        nrows=2, ncols=1, sharex=True, tooltip="Plot of local properties"
    )

    def __post_init__(self):
        # Initialize multi-plot canvas
        self.plot.min_height = 240
        self.plot[0].ylabel = "spacing (nm)"
        self.plot[0].legend.visible = False
        self.plot[0].border = [1, 1, 1, 0.2]
        self.plot[1].xlabel = "position (nm)"
        self.plot[1].ylabel = "skew (deg)"
        self.plot[1].legend.visible = False
        self.plot[1].border = [1, 1, 1, 0.2]

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
        self.plot[0].layers.clear()
        self.plot[1].layers.clear()
        return None

    def _plot_properties(self, spl: "CylSpline"):
        if (x := spl.props.get_loc(H.spl_dist, None)) is None:
            return None
        if x[0] > x[-1]:
            x = x[::-1]
        pitch_color = "lime"
        skew_color = "gold"

        self._init_plot()

        if (_interv := spl.props.get_loc(H.spacing, None)) is not None:
            self.plot[0].add_curve(x, _interv, color=pitch_color)
        if (_skew := spl.props.get_loc(H.skew, None)) is not None:
            self.plot[1].add_curve(x, _skew, color=skew_color)

        self.plot[0].xlim = (x[0] - 2, x[-1] + 2)
        self.plot[0].add_infline(
            pos=[x[0], 0], degree=90, color=[1.0, 0.0, 0.0, 0.3], lw=2
        )
        self.plot[1].add_infline(
            pos=[x[0], 0], degree=90, color=[1.0, 0.0, 0.0, 0.3], lw=2
        )
        self._plot_spline_position(x[0])
        return None

    def _plot_spline_position(self, x: float):
        if len(self.plot[0].layers) > 0:
            self.plot[0].layers[-1].pos = [x, 0]
        if len(self.plot[1].layers) > 0:
            self.plot[1].layers[-1].pos = [x, 0]
        xmin, xmax = self.plot[0].xlim
        if x < xmin or xmax < x:
            dx = xmax - xmin
            self.plot[0].xlim = (x - dx / 2, x + dx / 2)
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
