import os
import json
from typing import Union, TYPE_CHECKING
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel
import impy as ip
import macrokit as mk
from magicclass import magicclass, field, vfield, MagicTemplate
from magicclass.widgets import ConsoleTextEdit
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.utils import thread_worker
from acryo import Molecules

from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget

def json_encoder(obj):    
    """An enhanced encoder."""
    
    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dict(as_series=False)
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
        self = cls(**js)
        file_dir = Path(path).parent
        self.resolve_path(file_dir)
        return self
    
    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        from .widgets.widget_utils import resolve_path
        
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
    
    def to_json(self, path: str) -> None:
        """Save project as a json file."""
        with open(path, mode="w") as f:
            json.dump(self.dict(), f, indent=4, separators=(",", ": "), default=json_encoder)
        return None

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget", 
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> "CylindraProject":
        """Construct a project from a widget state."""
        from .widgets.widget_utils import get_versions
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
            template_image = as_relative(gui.sta.template_path),
            mask_parameters = gui.sta._get_mask_params(),
            tilt_range = gui.sta.tilt_range,
            macro = as_relative(macro_path),
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
            gui.sta.template_path = self.template_image or ""
            gui.sta._set_mask_params(self.mask_parameters)
            gui.reset_choices()
            gui._need_save = False
        
        return _load_project_on_return
    
    def make_project_viewer(self) -> "ProjectViewer":
        """Build a project viewer widget from this project."""
        pviewer = ProjectViewer()
        pviewer._from_project(self)
        return pviewer
    
    def make_molecules_viewer(self) -> "MoleculesViewer":
        """Build a molecules viewer widget from this project."""
        mviewer = MoleculesViewer()
        mviewer._from_project(self)
        return mviewer

@magicclass(labels=False, name="General info")
class Info(MagicTemplate):
    text = vfield(ConsoleTextEdit)
    
    def _from_project(self, project: CylindraProject):
        info = {
            "Date": project.datetime,
            "Version": project.version,
            "Dependencies": "<br>".join(
                map(lambda x: "{}={}".format(*x), project.dependency_versions.items())
            ),
            "Image": str(project.image),
            "Image scale": f"{project.scale} nm/pixel",
            "Multiscales": ", ".join(map(str, project.multiscales,)),
            "FT size": f"{project.current_ft_size:.1f} nm",
        }
        info_str = "<br>".join(map(lambda x: "<h2>{}</h2>{}".format(*x), info.items()))
        self.text = info_str
        self["text"].read_only = True  

@magicclass(labels=False, name="Splines")
class SplineViewer(MagicTemplate):
    canvas = field(Vispy3DCanvas)
    
    def _from_project(self, project: CylindraProject):
        from cylindra.components import CylSpline

        for path in project.splines:
            spl = CylSpline.from_json(path)
            coords = spl.partition(100)
            self.canvas.add_curve(coords, color="lime", width=3.0)
        
        # draw edge
        img = ip.lazy_imread(project.image)
        nz, ny, nx = img.shape
        for z in [0, nz]:
            arr = np.array([[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0], [z, 0, 0]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")
        for y, x in [(0, 0), (0, nx), (ny, nx), (ny, 0)]:
            arr = np.array([[0, y, x], [nz, y, x]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")
@magicclass(labels=False, widget_type="split")
class Properties(MagicTemplate):
    table_local = vfield([], widget_type="Table")
    table_global = vfield([], widget_type="Table")
    
    def _from_project(self, project: CylindraProject):
        if path := project.localprops:
            df = pd.read_csv(path)
            self.table_local = df
        self["table_local"].read_only = True
        
        if path := project.globalprops:
            df = pd.read_csv(path)
            self.table_global = df
        self["table_global"].read_only = True
        
@magicclass(labels=False, name="Molecules")
class MoleculesViewer(MagicTemplate):
    canvas = field(Vispy3DCanvas)
    
    def _from_project(self, project: CylindraProject):
        from acryo import Molecules

        for path in project.molecules:
            mole = Molecules.from_csv(path)
            self.canvas.add_points(mole.pos, face_color="lime")

        # draw edge
        img = ip.lazy_imread(project.image)
        nz, ny, nx = img.shape
        for z in [0, nz]:
            arr = np.array([[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0], [z, 0, 0]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")
        for y, x in [(0, 0), (0, nx), (ny, nx), (ny, 0)]:
            arr = np.array([[0, y, x], [nz, y, x]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")

@magicclass(labels=False, name="Global variables")
class GlobalVariables(MagicTemplate):
    text = vfield(str, widget_type=ConsoleTextEdit)
    
    def _from_project(self, project: CylindraProject):
        if path := project.global_variables:
            with open(path, mode="r") as f:
                self.text = f.read()
        self["text"].read_only = True

@magicclass(name="Subtomogram averaging")
class SubtomogramAveraging(MagicTemplate):
    template_image = field(Vispy3DCanvas)
    mask_parameters = vfield(str)
    tilt_range = vfield(str)
    
    def _from_project(self, project: CylindraProject):
        from skimage.filters.thresholding import threshold_yen
        
        if project.template_image is None or Path(project.template_image).is_dir():
            # no template image available
            pass
        else:
            img = ip.imread(project.template_image)
            thr = threshold_yen(img.value)
            self.template_image.add_image(img, rendering="iso", iso_threshold=thr)
        self.mask_parameters = str(project.mask_parameters)
        if project.tilt_range is not None:
            s0, s1 = project.tilt_range
            self.tilt_range = f"({s0:.1f}, {s1:.1f})"
@magicclass(labels=False)
class Macro(MagicTemplate):
    text = vfield(str, widget_type=ConsoleTextEdit)
    
    def _from_project(self, project: CylindraProject):
        if path := project.macro:
            with open(path, mode="r") as f:
                self.text = f.read()
        self["text"].read_only = True
        self["text"].syntax_highlight("python")

@magicclass(widget_type="tabbed", name="Project Viewer", record=False)
class ProjectViewer(MagicTemplate):
    info_viewer = field(Info)
    spline_viewer = field(SplineViewer)
    properties = field(Properties)
    molecules_viewer = field(MoleculesViewer)
    global_variables = field(GlobalVariables)
    subtomogram_averaging = field(SubtomogramAveraging)
    macro_viewer = field(Macro)
    
    def _from_project(self, project: CylindraProject):
        self.info_viewer._from_project(project)
        self.spline_viewer._from_project(project)
        self.properties._from_project(project)
        self.molecules_viewer._from_project(project)
        self.global_variables._from_project(project)
        self.subtomogram_averaging._from_project(project)
        self.macro_viewer._from_project(project)
