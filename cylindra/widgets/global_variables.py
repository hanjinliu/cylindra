from pathlib import Path
from typing import Tuple
import json
from magicclass import (
    magicmenu,
    set_options,
    set_design,
    MagicTemplate,
    get_function_gui,
)
from magicclass.utils import show_messagebox

from .widget_utils import FileFilter

from ..const import nm, GVar

INITIAL_PATH = Path(__file__).parent / "variables"

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
        daskChunk: Tuple[int, int, int] = GVar.daskChunk,
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
    
    @set_options(path={"filter": FileFilter.JSON})
    @set_design(text="Load variables")
    def load_variables(self, path: Path = INITIAL_PATH):
        """Load global variables from a Json file."""
        with open(path, mode="r") as f:
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
    
    @set_options(path={"filter": FileFilter.JSON, "mode": "w"})
    @set_design(text="Save variables")
    def save_variables(self, path: Path = INITIAL_PATH):
        """Save current global variables to a Json file."""
        gvar = GVar.get_value()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
        return None
