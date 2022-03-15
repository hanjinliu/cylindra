from pathlib import Path
import json
from magicclass import (
    magicmenu,
    set_options,
    MagicTemplate,
    )
from magicclass.types import Tuple as _Tuple
from magicclass import get_function_gui

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
        splError={"max": 5.0, "step": 0.1},
        inner={"step": 0.1},
        outer={"step": 0.1},
        fitLength={"min": 3.0, "max": 100.0},
        fitWidth={"min": 3.0, "max": 100.0},
        daskChunk={"options": {"min": 16, "max": 2048, "step": 16}},
        GPU={"label": "Use GPU if available"},
    )
    def Set_variables(
        self,
        nPFmin: int = GVar.nPFmin,
        nPFmax: int = GVar.nPFmax,
        splOrder: int = GVar.splOrder,
        yPitchMin: nm = GVar.yPitchMin,
        yPitchMax: nm = GVar.yPitchMax,
        minSkew: float = GVar.minSkew,
        maxSkew: float = GVar.maxSkew,
        splError: nm = GVar.splError,
        inner: float = GVar.inner,
        outer: float = GVar.outer,
        fitLength: nm = GVar.fitLength,
        fitWidth: nm = GVar.fitWidth,
        daskChunk: _Tuple[int, int, int] = GVar.daskChunk,
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
        splError : nm
            Average error of spline fitting.
        inner : float
            Radius x inner will be the inner surface of MT.
        outer : float
            Radius x outer will be the outer surface of MT.
        """        
        GVar.set_value(**locals())
    
    @set_options(path={"filter": FileFilter.JSON})
    def Load_variables(self, path: Path = INITIAL_PATH):
        with open(path, mode="r") as f:
            gvar = json.load(f)
        get_function_gui(self, "Set_variables")(**gvar, update_widget=True)
        return None
    
    @set_options(path={"filter": FileFilter.JSON, "mode": "w"})
    def Save_variables(self, path: Path = INITIAL_PATH):
        gvar = GVar.get_value()
        with open(path, mode="w") as f:
            json.dump(gvar, f, indent=4, separators=(", ", ": "))
        return None