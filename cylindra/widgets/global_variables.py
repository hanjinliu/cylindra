import json
from appdirs import user_config_dir
from typing import Annotated
import json

from magicclass import (
    magicmenu,
    set_options,
    set_design,
    MagicTemplate,
    nogui,
)
from magicclass.utils import show_messagebox
from magicclass.types import Path

from .widget_utils import FileFilter
from cylindra.const import nm, GlobalVariables as GVar

VAR_PATH = Path(user_config_dir("variables", "cylindra"))
SETTINGS_PATH = Path(user_config_dir("settings", "cylindra"))
USER_SETTINGS_NAME = "user-settings.json"
DEFAULT_VARIABLES = "default_variables"


@magicmenu(name="Global variables ...")
class GlobalVariablesMenu(MagicTemplate):
    def _get_file_names(self, *_) -> list[str]:
        try:
            return [fp.stem for fp in VAR_PATH.glob("*.json")]
        except Exception:
            return []

    @set_options(
        yPitchMin={"step": 0.1},
        yPitchMax={"step": 0.1},
        minSkew={"min": -90, "max": 90},
        maxSkew={"min": -90, "max": 90},
        minCurvatureRadius={"max": 10000.0},
        clockwise={"choices": ["MinusToPlus", "PlusToMinus"]},
        inner={"step": 0.1},
        outer={"step": 0.1},
        fitLength={"min": 3.0, "max": 100.0},
        fitWidth={"min": 3.0, "max": 100.0},
        daskChunk={"options": {"min": 16, "max": 2048, "step": 16}},
        GPU={"label": "Use GPU if available"},
    )
    @set_design(text="Set variables")
    def set_variables(
        self,
        nPFmin: int = GVar.nPFmin,
        nPFmax: int = GVar.nPFmax,
        splOrder: int = GVar.splOrder,
        yPitchMin: nm = GVar.yPitchMin,
        yPitchMax: nm = GVar.yPitchMax,
        minSkew: float = GVar.minSkew,
        maxSkew: float = GVar.maxSkew,
        minCurvatureRadius: nm = GVar.minCurvatureRadius,
        clockwise: str = GVar.clockwise,
        inner: float = GVar.inner,
        outer: float = GVar.outer,
        fitLength: nm = GVar.fitLength,
        fitWidth: nm = GVar.fitWidth,
        pointSize: float = GVar.pointSize,
        daskChunk: tuple[int, int, int] = GVar.daskChunk,
        GPU: bool = GVar.GPU,
    ):
        """
        Set global variables.

        Parameters
        ----------
        nPFmin : int
            Minimum protofilament numbers.
        nPFmax : int
            Maximum protofilament numbers.
        splOrder : int
            Maximum order of spline curve.
        yPitchMin : nm
            Minimum pitch length for estimation.
        yPitchMax : nm
            Maximum pitch length for estimation.
        minSkew : float
            Minimum skew angle for estimation.
        maxSkew : float
            Maximum skew angle for estimation.
        minCurvatureRadius : nm
            Minimum curvature radius of spline.
        inner : float
            Radius x inner will be the inner surface of the cylinder.
        outer : float
            Radius x outer will be the outer surface of the cylinder.
        """
        GVar.update(locals())

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
        self, name: Annotated[str, {"choices": _get_file_names}]
    ):
        """Load global variables from one of the saved Json files."""
        path = VAR_PATH / f"{name}.json"
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

    @set_design(text="Save variables")
    def save_variables(self, path: Path.Save[FileFilter.JSON] = VAR_PATH):
        """Save current global variables to a Json file."""
        gvar = GVar.dict()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
        return None

    @nogui
    def load_default(self):
        """Load default global variables."""

        with open(SETTINGS_PATH / USER_SETTINGS_NAME) as f:
            js = json.load(f)

        self.load_variables_by_name(js[DEFAULT_VARIABLES])
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
if not VAR_PATH.exists() or _is_empty(VAR_PATH):
    try:
        if not VAR_PATH.exists():
            VAR_PATH.mkdir(parents=True)

        _data_dir = Path(__file__).parent.parent / "_data"
        for fp in _data_dir.glob("*.json"):
            with open(fp) as f:
                js = json.load(f)

            with open(VAR_PATH / fp.name, mode="w") as f:
                json.dump(js, f, indent=4, separators=(", ", ": "))

    except Exception as e:
        print("Failed to initialize config directory.")
        print(e)
    else:
        print(f"Config directory initialized at {VAR_PATH}.")

if not SETTINGS_PATH.exists() or _is_empty(SETTINGS_PATH):
    try:
        if not SETTINGS_PATH.exists():
            SETTINGS_PATH.mkdir(parents=True)

        settings_js = {DEFAULT_VARIABLES: "eukaryotic_MT"}
        with open(SETTINGS_PATH / USER_SETTINGS_NAME, mode="w") as f:
            json.dump(settings_js, f, indent=4, separators=(", ", ": "))
    except Exception as e:
        print("Failed to initialize settings directory.")
        print(e)
    else:
        print(f"Settings directory initialized at {SETTINGS_PATH}.")
