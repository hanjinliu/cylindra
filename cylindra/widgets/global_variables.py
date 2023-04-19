import json
from appdirs import user_config_dir

from magicclass import (
    magicmenu,
    set_options,
    set_design,
    MagicTemplate,
    get_function_gui,
)
from magicclass.utils import show_messagebox
from magicclass.types import Path

from .widget_utils import FileFilter
from cylindra.const import nm, GlobalVariables as GVar

INITIAL_PATH = Path(user_config_dir("variables", "cylindra"))


@magicmenu(name="Global variables ...")
class GlobalVariables(MagicTemplate):
    @set_options(
        yPitchMin={"step": 0.1},
        yPitchMax={"step": 0.1},
        minSkew={"min": -90, "max": 90},
        maxSkew={"min": -90, "max": 90},
        minCurvatureRadius={"max": 10000.0},
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
        inner: float = GVar.inner,
        outer: float = GVar.outer,
        fitLength: nm = GVar.fitLength,
        fitWidth: nm = GVar.fitWidth,
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
        GVar.set_value(**locals())

    @set_design(text="Load variables")
    def load_variables(self, path: Path.Read[FileFilter.JSON] = INITIAL_PATH):
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

        get_function_gui(self, "set_variables")(**gvar, update_widget=True)
        return None

    @set_design(text="Save variables")
    def save_variables(self, path: Path.Save[FileFilter.JSON] = INITIAL_PATH):
        """Save current global variables to a Json file."""
        gvar = GVar.get_value()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
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
if not INITIAL_PATH.exists() or _is_empty(INITIAL_PATH):
    try:
        if not INITIAL_PATH.exists():
            INITIAL_PATH.mkdir(parents=True)

        import json

        eukaryotic_mt_gvar = {
            "nPFmin": 11,
            "nPFmax": 17,
            "splOrder": 3,
            "yPitchMin": 3.9,
            "yPitchMax": 4.3,
            "minSkew": -1.0,
            "maxSkew": 1.0,
            "minCurvatureRadius": 400.0,
            "inner": 0.8,
            "outer": 1.3,
            "fitLength": 48.0,
            "fitWidth": 44.0,
            "daskChunk": [256, 256, 256],
            "GPU": True,
        }

        tmv_gvar = {
            "nPFmin": 15,
            "nPFmax": 17,
            "splOrder": 3,
            "yPitchMin": 2.1,
            "yPitchMax": 2.5,
            "minSkew": -20.0,
            "maxSkew": -10.0,
            "minCurvatureRadius": 10000.0,
            "inner": 0.3,
            "outer": 1.5,
            "fitLength": 48.0,
            "fitWidth": 28.0,
            "daskChunk": [256, 256, 256],
            "GPU": True,
        }

        with open(INITIAL_PATH / "eukaryotic_microtubule.json", mode="w") as f:
            json.dump(eukaryotic_mt_gvar, f, indent=4, separators=(", ", ": "))
        with open(INITIAL_PATH / "TMV.json", mode="w") as f:
            json.dump(tmv_gvar, f, indent=4, separators=(", ", ": "))
    except Exception as e:
        print("Failed to initialize config directory.")
        print(e)
    else:
        print(f"Config directory initialized at {INITIAL_PATH}.")
