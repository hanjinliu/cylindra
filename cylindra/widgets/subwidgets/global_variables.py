import json
from typing import Annotated, Literal

from magicclass import (
    impl_preview,
    magicmenu,
    set_design,
    MagicTemplate,
    nogui,
)
from magicclass.types import Path
from magicclass.logging import getLogger
from cylindra.widgets.widget_utils import FileFilter
from cylindra.const import nm, GlobalVariables as GVar
from cylindra import _config

_Logger = getLogger("cylindra")


@magicmenu(name="Global variables ...", record=False)
class GlobalVariablesMenu(MagicTemplate):
    def _get_file_names(self, *_) -> list[str]:
        try:
            return [fp.stem for fp in _config.VAR_PATH.glob("*.json")]
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
        rise_sign: Literal[-1, 1] = -1,
        min_curvature_radius: Annotated[float, {"max": 1e4}] = 100,
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
        rise_sign : -1 or 1
            Sign of rise angle.
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
        dask_chunk : (int, int, int)
            Chunk size for dask array.
        use_gpu : bool
            Use GPU if available.
        """
        loc = locals()
        loc.pop("self")
        return self.update_variables(**loc)

    @nogui
    def load_variables(self, path: "Path | str"):
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
            _Logger.warning(
                "Could not load following variables, maybe due to version "
                f"incompatibility: {_undef!r}"
            )
        self.set_variables(**gvar)
        _Logger.print(f"Global variables are set to:\n{Path(path).as_posix()}")
        return None

    @nogui
    def update_variables(
        self,
        npf_min: "int | None" = None,
        npf_max: "int | None" = None,
        spline_degree: "int | None" = None,
        spacing_min: "float | None" = None,
        spacing_max: "float | None" = None,
        skew_min: "float | None" = None,
        skew_max: "float | None" = None,
        rise_sign: Literal[-1, 1] = -1,
        min_curvature_radius: "float | None" = None,
        clockwise: Literal["MinusToPlus", "PlusToMinus", None] = None,
        thickness_inner: "float | None" = None,
        thickness_outer: "float | None" = None,
        fit_depth: "float | None" = None,
        fit_width: "float | None" = None,
        point_size: "float | None" = None,
        dask_chunk: "tuple[int, int, int] | None" = None,
        use_gpu: "bool | None" = None,
    ):
        loc = locals()
        loc.pop("self")
        for k, v in list(loc.items()):
            if v is None:
                loc.pop(k)
        GVar.update(loc)
        return GVar

    @set_design(text="Load variables")
    def load_variables_by_name(
        self,
        var_name: Annotated[str, {"choices": _get_file_names, "label": "Variable set"}],
    ):
        """Load global variables from one of the saved Json files."""
        path = _config.VAR_PATH / f"{var_name}.json"
        self.load_variables(path)
        return None

    @set_design(text="Save variables")
    def save_variables(self, path: Path.Save[FileFilter.JSON] = _config.VAR_PATH):
        """Save current global variables to a Json file."""
        gvar = GVar.dict()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
        return None

    @nogui
    def load_default(self):
        """Load default global variables."""

        with open(_config.USER_SETTINGS) as f:
            js = json.load(f)

        self.load_variables_by_name(js[_config.DEFAULT_VARIABLES])
        return None


@impl_preview(GlobalVariablesMenu.load_variables_by_name)
def _(self: GlobalVariablesMenu, var_name: str):
    from cylindra.widgets._previews import view_text
    from cylindra.widgets import CylindraMainWidget

    path = _config.VAR_PATH / f"{var_name}.json"
    wdt = view_text(path, parent=self)
    self.find_ancestor(CylindraMainWidget)._active_widgets.add(wdt)
    return None
