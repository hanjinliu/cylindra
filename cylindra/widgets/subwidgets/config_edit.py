from typing import Annotated

from magicclass import abstractapi, field, magicclass, nogui, set_design
from magicgui.types import Separator

from cylindra import _config
from cylindra.components.spline import SplineConfig
from cylindra.widgets._annotated import SplinesType
from cylindra.widgets.subwidgets._child_widget import ChildWidget


def _get_saved_config_files(self, w=None) -> list[str]:
    return [path.stem for path in _config.get_config().list_config_paths()]


@magicclass(widget_type="groupbox")
class ConfigContainer(ChildWidget):
    """Widget to edit the spline config.

    Attributes
    ----------
    npf_range : tuple[int, int]
        Min/max number of protofilaments.
    spacing_range : tuple[float, float]
        Min/max longitudinal lattice spacing (nm).
    twist_range : tuple[float, float]
        Min/max helical twist (degrees per subunit).
    rise_range : tuple[float, float]
        Min/max rise angle (degrees).
    rise_sign : int
        Sign of the rise angle, either -1 or 1. This parameter is used to make the start
        number positive. Practically, this parameter is determined experimentally during
        analysis.
    clockwise : str
        Specifies whether the clockwise swirl of the cylinder cross section corresponds
        to the "PlusToMinus" or "MinusToPlus" polarity. This parameter is used for the
        polarity inference.
    thickness_inner : float
        Inner thickness (distance between the inner edge and the peak of the radial
        profile) of the cylinder (nm).
    thickness_outer : float
        Outer thickness (distance between the outer edge and the peak of the radial
        profile) of the cylinder (nm).
    fit_depth : float
        Depth of the box used for the spline fitting tasks (nm).
    fit_width : float
        Width of the box used for the spline fitting tasks (nm).
    """

    npf_range = field((11, 17)).with_options(options={"min": 2, "max": 100})
    spacing_range = field((3.9, 4.3)).with_options(options={"step": 0.05})
    twist_range = field((-1.0, 1.0)).with_options(
        options={"min": -45.0, "max": 45.0, "step": 0.05}
    )
    rise_range = field((0.0, 45.0)).with_options(
        options={"min": -45.0, "max": 45.0, "step": 0.1}
    )
    rise_sign = field(-1).with_choices([-1, 1])
    clockwise = field("MinusToPlus").with_choices(["PlusToMinus", "MinusToPlus"])
    thickness_inner = field(2.8).with_options(min=0.0, step=0.1)
    thickness_outer = field(2.8).with_options(min=0.0, step=0.1)
    fit_depth = field(49.0).with_options(min=4.0, step=1)
    fit_width = field(44.0).with_options(min=4.0, step=1)

    sep0 = Separator

    @set_design(text="Load preset")
    def load_config_preset(
        self, name: Annotated[str, {"choices": _get_saved_config_files}]
    ):
        """Load a preset config file as the default config."""
        path = _config.get_config().spline_config_path(name)
        config = SplineConfig.from_file(path)
        self.set_config(config)

    @set_design(text="Save as new preset")
    def save_default_config(self, name: str):
        """Save current default config as a preset."""
        path = _config.get_config().spline_config_path(name)
        if path.exists():
            raise FileExistsError(f"Config file {path} already exists.")
        self.get_config().to_file(path)
        return self._get_main().reset_choices()

    @set_design(text="Refer current spline config")
    def refer_current_spline_config(self):
        """Refer the config of the current spline to this widget."""
        main = self._get_main()
        idx = main._get_spline_idx()
        config = main.splines[idx].config
        self.set_config(config)

    @set_design(text="Apply to splines")
    def apply_to_splines(self, splines: SplinesType = None):
        """Overwrite the config of the specified splines with the edited config."""
        main = self._get_main()
        config_new = self.get_config()
        main.update_spline_config(splines, **config_new.asdict())

    @nogui
    def get_config(self) -> SplineConfig:
        return SplineConfig().updated(
            npf_range=self.npf_range.value,
            spacing_range=self.spacing_range.value,
            twist_range=self.twist_range.value,
            rise_range=self.rise_range.value,
            rise_sign=self.rise_sign.value,
            clockwise=self.clockwise.value,
            thickness_inner=self.thickness_inner.value,
            thickness_outer=self.thickness_outer.value,
            fit_depth=self.fit_depth.value,
            fit_width=self.fit_width.value,
        )

    @nogui
    def set_config(self, config: SplineConfig):
        with self.changed.blocked():
            self.npf_range.value = config.npf_range.astuple()
            self.spacing_range.value = config.spacing_range.astuple()
            self.twist_range.value = config.twist_range.astuple()
            self.rise_range.value = config.rise_range.astuple()
            self.rise_sign.value = config.rise_sign
            self.clockwise.value = config.clockwise
            self.thickness_inner.value = config.thickness_inner
            self.thickness_outer.value = config.thickness_outer
            self.fit_depth.value = config.fit_depth
            self.fit_width.value = config.fit_width
        self.changed.emit()


@magicclass(labels=False, record=False)
class ConfigEdit(ChildWidget):
    @magicclass(layout="horizontal")
    class Header(ChildWidget):
        config_current = field(ConfigContainer, name="Current default config")

        @magicclass
        class Buttons(ChildWidget):
            left = abstractapi()
            right = abstractapi()

        @set_design(text="<", location=Buttons)
        def left(self):
            """Copy edited value to the current default config."""
            self.config_current.set_config(self.config_new.get_config())

        @set_design(text=">", location=Buttons)
        def right(self):
            """Copy current default config to the edit field."""
            self.config_new.set_config(self.config_current.get_config())

        config_new = field(ConfigContainer, name="Edit field")

        def __post_init__(self):
            self.config_current.npf_range.enabled = False
            self.config_current.spacing_range.enabled = False
            self.config_current.twist_range.enabled = False
            self.config_current.rise_range.enabled = False
            self.config_current.rise_sign.enabled = False
            self.config_current.clockwise.enabled = False
            self.config_current.thickness_inner.enabled = False
            self.config_current.thickness_outer.enabled = False
            self.config_current.fit_depth.enabled = False
            self.config_current.fit_width.enabled = False

            @self.config_current.changed.connect
            def _on_current_changed(*_):
                main = self._get_main()
                main._refer_spline_config(self.config_current.get_config())

    @property
    def config_current(self) -> ConfigContainer:
        return self.Header.config_current

    @property
    def config_new(self) -> ConfigContainer:
        return self.Header.config_new
