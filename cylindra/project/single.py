import os
import json
from typing import Union, TYPE_CHECKING
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import macrokit as mk
from acryo import Molecules

from cylindra.const import PropertyNames as H, get_versions
from ._base import BaseProject, PathLike

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget


class CylindraProject(BaseProject):
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
    project_path: Union[Path, None] = None
    
    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.image = resolve_path(self.image, file_dir)
        self.localprops = resolve_path(self.localprops, file_dir)
        self.globalprops = resolve_path(self.globalprops, file_dir)
        self.template_image = resolve_path(self.template_image, file_dir)
        self.global_variables = resolve_path(self.global_variables, file_dir)
        self.splines = [resolve_path(p, file_dir) for p in self.splines]
        self.molecules = [resolve_path(p, file_dir) for p in self.molecules]
        self.macro = resolve_path(self.macro, file_dir)
        return self

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget", 
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> "CylindraProject":
        """Construct a project from a widget state."""
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
        molecules_paths: list[Path] = []
        for layer in filter(
            lambda x: isinstance(x, Points) and MOLECULES in x.metadata,
            gui.parent_viewer.layers
        ):
            molecules_paths.append((results_dir/layer.name).with_suffix(".csv"))
        
        # Save path of  global variables
        gvar_path = results_dir / "global_variables.json"
        
        # Save path of macro
        macro_path = results_dir / "script.py"
        
        from datetime import datetime
        
        file_dir = json_path.parent
        def as_relative(p: Path):
            try:
                out = p.relative_to(file_dir)
            except Exception:
                out = p
            return out

        self = cls(
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
            template_image = as_relative(gui.sta.params.template_path),
            mask_parameters = gui.sta.params._get_mask_params(),
            tilt_range = gui.sta.params.tilt_range,
            macro = as_relative(macro_path),
            project_path=json_path,
        )
        return self
        
        
    @classmethod
    def save_gui(
        cls: "CylindraProject",
        gui: "CylindraMainWidget", 
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> None:
        """
        Serialize the GUI state to a json file.
        
        Parameters
        ----------
        gui : CylindraMainWidget
            The main widget from which project model will be constructed.
        json_path : Path
            The path to the project json file.
        results_dir : Path, optional
            The directory to save the results.
        """
        from napari.layers import Points
        from cylindra.const import MOLECULES
        
        self = cls.from_gui(gui, json_path, results_dir)
        # save objects
        self.to_json(json_path)
        macro_str = str(gui._format_macro(gui.macro[gui._macro_offset:]))
        
        tomo = gui.tomogram
        localprops = tomo.collect_localprops()    
        globalprops = tomo.collect_globalprops()
        
        if results_dir is None:
            results_dir = json_path.parent / (json_path.stem + "_results")
        else:
            results_dir = Path(results_dir)
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = None if globalprops is None else results_dir / "globalprops.csv"
        
        molecule_dataframes: list[pl.DataFrame] = []
        molecules_paths: list[Path] = []
        for layer in filter(
            lambda x: isinstance(x, Points) and MOLECULES in x.metadata,
            gui.parent_viewer.layers
        ):
            layer: Points
            mole: "Molecules" = layer.metadata[MOLECULES]
            molecule_dataframes.append(mole.to_dataframe())
            molecules_paths.append((results_dir/layer.name).with_suffix(".csv"))
        
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # create a directory if not exists.
        if localprops_path:
            localprops.to_csv(localprops_path)
        if globalprops_path:
            globalprops.to_csv(globalprops_path)
        if self.splines:
            for spl, path in zip(gui.tomogram.splines, self.splines):
                spl.to_json(results_dir.parent / path)
        if self.molecules:
            for df, path in zip(molecule_dataframes, self.molecules):
                df.write_csv(results_dir.parent / path)
        
        gui.Others.Global_variables.save_variables(results_dir.parent / self.global_variables)
        
        if macro_str:
            with open(results_dir.parent / self.macro, mode="w") as f:
                f.write(macro_str)
        return None
    
    def to_gui(self, gui: "CylindraMainWidget", filter: bool = True):
        from cylindra.components import CylSpline, CylTomogram
        from magicclass.utils import thread_worker
        
        gui.tomogram = CylTomogram.imread(
            path=self.image, 
            scale=self.scale, 
            binsize=self.multiscales, 
        )
        
        gui._current_ft_size = self.current_ft_size
        gui._macro_offset = len(gui.macro)
        
        # load splines
        splines = [CylSpline.from_json(path) for path in self.splines]
        localprops_path = self.localprops
        if localprops_path is not None:
            all_localprops = dict(iter(pd.read_csv(localprops_path).groupby("SplineID")))
        else:
            all_localprops = {}
        globalprops_path = self.globalprops
        if globalprops_path is not None:
            all_globalprops = dict(pd.read_csv(globalprops_path, index_col=0).iterrows())
        else:
            all_globalprops = {}
        
        for i, spl in enumerate(splines):
            spl.localprops = all_localprops.get(i, None)
            if spl.localprops is not None:
                spl._anchors = np.asarray(spl.localprops.get(H.splPosition))
                spl.localprops.pop("SplineID")
                spl.localprops.pop("PosID")
                spl.localprops.index = range(len(spl.localprops))
            spl.globalprops = all_globalprops.get(i, None)
            if spl.globalprops is not None:
                try:
                    spl.radius = spl.globalprops.pop("radius")
                except KeyError:
                    pass
                try:
                    spl.orientation = spl.globalprops.pop("orientation")
                except KeyError:
                    pass
        
        @thread_worker.to_callback
        def _load_project_on_return():
            gui._send_tomogram_to_viewer(filt=filter)
            
            if splines:
                gui.tomogram._splines = splines
                gui._update_splines_in_images()
                with gui.macro.blocked():
                    gui.sample_subtomograms()
            
            # load molecules
            for path in self.molecules:
                mole = Molecules.from_csv(path)
                gui.add_molecules(mole, name=Path(path).stem)
            
            # load global variables
            if self.global_variables:
                with gui.macro.blocked():
                    gui.Others.Global_variables.load_variables(self.global_variables)
            
            # append macro
            with open(self.macro) as f:
                txt = f.read()
                
            macro = mk.parse(txt)
            gui.macro.extend(macro.args)

            # load subtomogram analyzer state
            gui.sta.params.template_path = self.template_image or ""
            gui.sta._set_mask_params(self.mask_parameters)
            gui.reset_choices()
            gui._need_save = False
        
        return _load_project_on_return
    
    def make_project_viewer(self):
        """Build a project viewer widget from this project."""
        from ._widgets import ProjectViewer
        pviewer = ProjectViewer()
        pviewer._from_project(self)
        return pviewer
    
    def make_component_viewer(self):
        from ._widgets import ComponentsViewer
        """Build a molecules viewer widget from this project."""
        mviewer = ComponentsViewer()
        mviewer._from_project(self)
        return mviewer


def resolve_path(path: Union[str, Path, None], root: Path) -> Union[Path, None]:
    """Resolve a relative path to an absolute path."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    path_joined = root / path
    if path_joined.exists():
        return path_joined
    raise ValueError(f"Path {path} could not be resolved under root path {root}.")
