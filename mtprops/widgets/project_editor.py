from pathlib import Path
from typing import Dict, List, Tuple
from magicclass import magicclass, magicmenu, vfield, MagicTemplate, do_not_record
from magicclass.qthreading import thread_worker
from magicgui.widgets import ProgressBar
from magicclass.types import Bound
import numpy as np
import impy as ip

from .widget_utils import FileFilter, get_versions
from .project import MTPropsProject, SubtomogramAveragingProject
from ..components.molecules import Molecules

@magicclass(widget_type="frame", labels=False)
class SubProject(MagicTemplate):
    subproject = vfield(Path, options={"filter": FileFilter.JSON, "tooltip": "Sub-project that will be used as a part of subtomogram averaging project."}, record=False)
    image = vfield(str, options={"enabled": False}, record=False)
    select = vfield(widget_type="Select", record=False)
    
    @subproject.connect
    def _update(self):
        try:
            project = MTPropsProject.from_json(self.subproject)
            molecule_paths = project.molecules
            image_path = project.image
        except Exception:
            molecule_paths = [] 
            image_path = ""
        self["select"].choices = molecule_paths or []
        self.image = image_path
    
    @property
    def value(self) -> Tuple[str, List[str]]:
        return (str(self.subproject), self.select)
    
    @value.setter
    def value(self, v: Tuple[str, List[str]]):
        subproject_path = str(v[0])
        self.subproject = subproject_path
        self["subproject"].changed.emit(subproject_path)
        self["select"].value = v[1]

N_BATCH = 24

@magicclass
class SubtomogramAveragingProjectEditor(MagicTemplate):
    @magicmenu
    class Menu(MagicTemplate):
        def Load(self): ...
        def Save(self): ...
    
    def __post_init__(self):
        self.add_dataset()
        self.min_width = 580
    
    def _get_my_path(self, _=None) -> Path:
        path = Path(self.project_path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return path

    project_path = vfield(Path, options={"mode": "w", "filter": FileFilter.JSON})
    
    @magicclass(widget_type="scrollable")
    class DataSets(MagicTemplate):
        pass
    
    @Menu.wraps
    def Load(self, path: Path):
        self.DataSets.clear()
        project = SubtomogramAveragingProject.from_json(path)
        for k, v in project.datasets.items():
            wdt = SubProject()
            self.DataSets.append(wdt)
            wdt.value = (k, v)
        self.project_path = path
        return project
    
    @Menu.wraps
    def Save(self, path: Bound[project_path]):
        self.project.to_json(path)
        self.project_path = path
        return None
    
    @do_not_record
    def add_dataset(self):
        wdt = SubProject()
        self.DataSets.append(wdt)
        return None
    
    @property
    def project(self) -> SubtomogramAveragingProject:
        datasets: Dict[str, List[str]] = {}
        shape_nm = np.zeros(3)  # initialize
        chunksize = 99999  # initialize
        for w in self.DataSets:
            w: SubProject
            subproject_path, molecules_path = w.value
            datasets[subproject_path] = molecules_path
            subproject = MTPropsProject.from_json(subproject_path)
            # use smaller chunk size
            chunksize = min(chunksize, subproject.chunksize)
            # use larger template shape
            template = ip.lazy_imread(subproject.template_image)
            shape_nm = np.maximum(
                shape_nm, 
                np.array(template.scale) * np.array(template.shape)
            )
            
        
        from datetime import datetime
        _versions = get_versions()
        return SubtomogramAveragingProject(
            datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version = _versions.pop("MTProps"),
            dependency_versions = _versions,
            datasets = datasets,
            shape = tuple(shape_nm),
            chunksize=chunksize,
        )
    
    def _get_n_iter(self):
        project = SubtomogramAveragingProject.from_json(self._get_my_path())
        niter = 0
        for molecules in project.datasets.values():
            allmole = Molecules.concat([Molecules.from_csv(mp) for mp in molecules])
            niter += len(allmole)
        return int(np.ceil(niter / N_BATCH))
    
    @thread_worker(progress={"desc": "Progress", "total": _get_n_iter})
    def run(self, path: Bound[_get_my_path], order: int = 1) -> ip.ImgArray:
        # TODO: how to deal with images with different scale?
        from ..components import MtTomogram
        project = SubtomogramAveragingProject.from_json(path)
        sum_img = 0
        total_size = 0
        for proj_path, molecules in project.datasets.items():
            subproject = MTPropsProject.from_json(proj_path)
            tomo = MtTomogram.imread(subproject.image, scale=subproject.scale)
            allmole = Molecules.concat([Molecules.from_csv(mp) for mp in molecules])
            loader = tomo.get_subtomogram_loader(
                mole=allmole,
                shape=project.shape,
                order=order,
                chunksize=subproject.chunksize
            )
            ave = yield from loader.iter_average(nbatch=N_BATCH)
            size = len(loader.molecules)
            sum_img += ave * size
            total_size += size
            
        sum_img.set_scale(ave)
        return sum_img / total_size
    
    @run.returned.connect
    def _on_returned(self, img: ip.ImgArray):
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        parent._subtomogram_averaging._show_reconstruction(img, "All average")
        return None

    pbar = vfield(ProgressBar, options={"visible": False})