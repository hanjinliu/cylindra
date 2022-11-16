import os
import json
from typing import Union, TYPE_CHECKING
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from .main import CylindraMainWidget
    from acryo import Molecules

def json_encoder(obj):    
    """An enhanced encoder."""
    
    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        if obj.is_absolute():
            return str(obj)
        else:
            return os.path.join(".", str(obj))
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")


PathLike = Union[Path, str]

class CylindraProject(BaseModel):
    """A project of cylindra."""
    
    datetime: str
    version: str
    dependency_versions: dict[str, str]
    image: PathLike
    scale: float
    multiscales: list[int]
    current_ft_size: float
    splines: list[PathLike]
    localprops: Union[PathLike, None]
    globalprops: Union[PathLike, None]
    molecules: list[PathLike]
    global_variables: PathLike
    template_image: Union[PathLike, None]
    mask_parameters: Union[None, tuple[float, float], PathLike]
    tilt_range: Union[tuple[float, float], None]
    macro: PathLike

    @classmethod
    def from_json(cls, path: str):
        """Construct a project from a json file."""
        path = str(path)
    
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        return cls(**js)
    
    def to_json(self, path: str) -> None:
        """Save project as a json file."""
        with open(path, mode="w") as f:
            json.dump(self.dict(), f, indent=4, separators=(",", ": "), default=json_encoder)
        return None

    @classmethod
    def save_gui(
        cls,
        gui: "CylindraMainWidget", 
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> None:
        from .widget_utils import get_versions
        from napari.layers import Points
        from cylindra.const import MOLECULES
        
        if json_path.suffix == "":
            json_path = json_path.with_suffix(".json")
        
        _versions = get_versions()
        tomo = gui.tomogram
        localprops = tomo.collect_localprops()    
        globalprops = tomo.collect_globalprops()
        
        if results_dir is None:
            results_dir = json_path.parent / (json_path.stem + "_results")
        else:
            results_dir = Path(results_dir)
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = None if globalprops is None else results_dir / "globalprops.csv"
        
        # Save path of splines
        spline_paths: list[Path] = []
        for i, spl in enumerate(gui.tomogram.splines):
            spline_paths.append(results_dir/f"spline-{i}.json")
            
        # Save path of molecules
        molecule_dataframes: list[pd.DataFrame] = []
        molecules_paths: list[Path] = []
        for layer in filter(
            lambda x: isinstance(x, Points) and MOLECULES in x.metadata,
            gui.parent_viewer.layers
        ):
            layer: Points
            mole: "Molecules" = layer.metadata[MOLECULES]
            molecule_dataframes.append(mole.to_dataframe())
            molecules_paths.append((results_dir/layer.name).with_suffix(".csv"))
        
        # Save path of  global variables
        gvar_path = results_dir / "global_variables.json"
        
        # Save path of macro
        macro_path = results_dir / "script.py"
        macro_str = str(gui._format_macro(gui.macro[gui._macro_offset:]))
        
        from datetime import datetime
        
        file_dir = json_path.parent
        def as_relative(p: Path):
            try:
                out = p.relative_to(file_dir)
            except Exception:
                out = p
            return out

        project = cls(
            datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version = next(iter(_versions.values())),
            dependency_versions = _versions,
            image = as_relative(tomo.source),
            scale = tomo.scale,
            multiscales = [x[0] for x in tomo.multiscaled],
            current_ft_size = gui._current_ft_size,
            splines = [as_relative(p) for p in spline_paths],
            localprops = as_relative(localprops_path),
            globalprops = as_relative(globalprops_path),
            molecules = [as_relative(p) for p in molecules_paths],
            global_variables = as_relative(gvar_path),
            template_image = as_relative(gui._subtomogram_averaging.template_path),
            mask_parameters = gui._subtomogram_averaging._get_mask_params(),
            tilt_range = gui._subtomogram_averaging.tilt_range,
            macro = as_relative(macro_path),
        )
        
        # save objects
        project.to_json(json_path)
        
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # create a directory if not exists.
        if localprops_path:
            localprops.to_csv(localprops_path)
        if globalprops_path:
            globalprops.to_csv(globalprops_path)
        if spline_paths:
            for spl, path in zip(gui.tomogram.splines, spline_paths):
                spl.to_json(path)
        if molecules_paths:
            for df, fp in zip(molecule_dataframes, molecules_paths):
                df.to_csv(fp, index=False)
        
        gui.Others.Global_variables.save_variables(gvar_path)
        
        if macro_str:
            with open(macro_path, mode="w") as f:
                f.write(macro_str)
        return None