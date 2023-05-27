import json
from appdirs import user_config_dir
from typing import Annotated, Literal
import json

from magicclass import (
    impl_preview,
    magicmenu,
    set_design,
    MagicTemplate,
    nogui,
)
from magicclass.utils import show_messagebox
from magicclass.types import Path

from .widget_utils import FileFilter
from cylindra.const import nm, GlobalVariables as GVar, ConfigConst as Cfg


@magicmenu(name="Global variables ...")
class GlobalVariablesMenu(MagicTemplate):
    def _get_file_names(self, *_) -> list[str]:
        try:
            return [fp.stem for fp in Cfg.VAR_PATH.glob("*.json")]
        except Exception:
            return []

    @set_design(text="Set variables")
    def set_variables(
        self,
        npf_min: Annotated[int, {"min": 1}] = 1,
        npf_max: Annotated[int, {"min": 1}] = 1,
        spline_degree: Annotated[int, {"min": 1, "max": 5}] = 3,
        spacing_min: Annotated[nm, {"step": 0.1}] = 1,
        spacing_max: Annotated[nm, {"step": 0.1}] = 2,
        skew_min: Annotated[float, {"min": -90, "max": 90}] = -1,
        skew_max: Annotated[float, {"min": -90, "max": 90}] = 1,
        min_curvature_radius: Annotated[float, {"max": 1e4}] = 100,
        deconv_range: Annotated[int, {"max": 100}] = 2,
        clockwise: Literal["MinusToPlus", "PlusToMinus"] = "MinusToPlus",
        thickness_inner: Annotated[nm, {"step": 0.1}] = 1.0,
        thickness_outer: Annotated[nm, {"step": 0.1}] = 1.0,
        fit_depth: Annotated[nm, {"step": 0.1}] = 10.0,
        fit_width: Annotated[nm, {"step": 0.1}] = 10.0,
        point_size: Annotated[nm, {"step": 0.1}] = 1.0,
        dask_chunk: Annotated[tuple[int, int, int], {"options": {"min": 16, "max": 2048, "step": 16}}] = (32, 32, 32),
        use_gpu: bool = True,
    ):  # fmt: skip
        """
        Set global variables.

        Parameters
        ----------
        npf_min : int
            Minimum protofilament numbers.
        npf_max : int
            Maximum protofilament numbers.
        spline_degree : int
            Maximum order of spline curve.
        spacing_min : nm
            Minimum pitch length for estimation.
        spacing_max : nm
            Maximum pitch length for estimation.
        skew_min : float
            Minimum skew angle for estimation.
        skew_max : float
            Maximum skew angle for estimation.
        min_curvature_radius : nm
            Minimum curvature radius of spline.
        clockwise : str
            Orientation to which clockwise rotation of the cylinder corresponds.
        thickness_inner : float
            Radius x inner will be the inner surface of the cylinder.
        thickness_outer : float
            Radius x outer will be the outer surface of the cylinder.
        fit_depth : nm
            Depth in nm of image that will be used for spline fitting.
        fit_width : nm
            Width in nm of image that will be used for spline fitting.
        point_size : float
            Default size of points layer in nm.
        dask_chunk : tuple[int, int, int]
            Chunk size for dask array.
        use_gpu : bool
            Use GPU if available.
        """
        loc = locals()
        loc.pop("self")
        return GVar.update(loc)

    @nogui
    def load_variables(self, path):
        """Load global variables from a Json file."""
        with open(path) as f:
            gvar: dict = json.load(f)

        # for version compatibility
        annots = GVar.__annotations__.keys()
        _undef = set()
        for k in gvar.keys():
            if k not in annots:
                _undef.add(k)
        if _undef:
            for k in _undef:
                gvar.pop(k)
            show_messagebox(
                mode="warn",
                title="Warning",
                text=(
                    "Could not load following variables, maybe due to version "
                    f"incompatibility: {_undef!r}"
                ),
                parent=self.native,
            )

        GVar.update(gvar)
        return None

    @set_design(text="Load variables")
    def load_variables_by_name(
        self,
        var_name: Annotated[str, {"choices": _get_file_names, "label": "Variable set"}],
    ):
        """Load global variables from one of the saved Json files."""
        path = Cfg.VAR_PATH / f"{var_name}.json"
        self.load_variables(path)
        return None

    @impl_preview(load_variables_by_name)
    def _(self, var_name: str):
        from ._previews import view_text
        from cylindra.widgets import CylindraMainWidget

        path = Cfg.VAR_PATH / f"{var_name}.json"
        wdt = view_text(path, parent=self)
        self.find_ancestor(CylindraMainWidget)._active_widgets.add(wdt)
        return None

    @set_design(text="Save variables")
    def save_variables(self, path: Path.Save[FileFilter.JSON] = Cfg.VAR_PATH):
        """Save current global variables to a Json file."""
        gvar = GVar.dict()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
        return None

    @nogui
    def load_default(self):
        """Load default global variables."""

        with open(Cfg.SETTINGS_PATH / Cfg.USER_SETTINGS_NAME) as f:
            js = json.load(f)

        self.load_variables_by_name(js[Cfg.DEFAULT_VARIABLES])
        return None


def _is_empty(path: Path) -> bool:
    """Check if a directory is empty."""
    it = path.glob("*")
    try:
        next(it)
    except StopIteration:
        return True
    return False


# Initialize user config directory.
if not Cfg.VAR_PATH.exists() or _is_empty(Cfg.VAR_PATH):  # pragma: no cover
    try:
        if not Cfg.VAR_PATH.exists():
            Cfg.VAR_PATH.mkdir(parents=True)

        _data_dir = Path(__file__).parent.parent / "_data"
        for fp in _data_dir.glob("*.json"):
            with open(fp) as f:
                js = json.load(f)

            with open(Cfg.VAR_PATH / fp.name, mode="w") as f:
                json.dump(js, f, indent=4, separators=(", ", ": "))

    except Exception as e:
        print("Failed to initialize config directory.")
        print(e)
    else:
        print(f"Config directory initialized at {Cfg.VAR_PATH}.")

if not Cfg.SETTINGS_PATH.exists() or _is_empty(Cfg.SETTINGS_PATH):  # pragma: no cover
    try:
        if not Cfg.SETTINGS_PATH.exists():
            Cfg.SETTINGS_PATH.mkdir(parents=True)

        settings_js = {Cfg.DEFAULT_VARIABLES: "eukaryotic_MT"}
        with open(Cfg.SETTINGS_PATH / Cfg.USER_SETTINGS_NAME, mode="w") as f:
            json.dump(settings_js, f, indent=4, separators=(", ", ": "))
    except Exception as e:
        print("Failed to initialize settings directory.")
        print(e)
    else:
        print(f"Settings directory initialized at {Cfg.SETTINGS_PATH}.")
