from pathlib import Path
from typing import Dict, List, Tuple
from magicclass import magicclass, vfield, MagicTemplate, do_not_record
import numpy as np
import impy as ip

from .widget_utils import FileFilter, get_versions
from .project import MTPropsProject, SubtomogramAveragingProject
from ..components.molecules import Molecules

@magicclass(labels=False)
class SubProject(MagicTemplate):
    subproject = vfield(Path, options={"filter": FileFilter.JSON, "tooltip": "Sub-project that will be used as a part of subtomogram averaging project."}, record=False)
    image = vfield(str, enabled=False, record=False)
    select = vfield(widget_type="Select", record=False)
    
    @subproject.connect
    def _update(self):
        try:
            project = MTPropsProject.from_json(self.subproject)
            molecule_paths = project.molecules
            
        except Exception:
            molecule_paths = [] 
        self["select"].choices = molecule_paths
        self.image = project.image
    
    @property
    def value(self) -> Tuple[str, List[str]]:
        return (self.subproject, self.select)


@magicclass
class SubtomogramAveragingProjectEditor(MagicTemplate):
    def __post_init__(self):
        self.add_dataset()
    
    @magicclass(widget_type="list")
    class DataSets(MagicTemplate):
        pass
    
    @do_not_record
    def add_dataset(self):
        wdt = SubProject()
        self.DataSets.append(wdt)
    
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
            shape = shape_nm,
            chunksize=chunksize,
        )
    
    def run(self, order: int = 1) -> ip.ImgArray:
        # TODO: how to deal with images with different scale?
        from ..components import MtTomogram
        project = self.project
        sum_img = 0
        total_size = 0
        for proj_path, molecules in project.datasets.items():
            subproject = MTPropsProject.from_json(proj_path)
            tomo = MtTomogram.imread(subproject.image, scale=subproject.scale)
            allmole = Molecules.concat([Molecules.from_csv(mp) for mp in molecules])
            loader = tomo.get_subtomogram_loader(
                mole=allmole,
                shape=project.shape,
                chunksize=subproject.chunksize
            )
            ave = loader.average(order=order)
            size = len(loader.molecules)
            sum_img += ave * size
            total_size += size
        sum_img.set_scale()
        return sum_img / total_size

    