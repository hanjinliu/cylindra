from typing import Iterator, Union, TYPE_CHECKING, Annotated
from timeit import default_timer
from macrokit import Expr
from magicclass import (
    magicclass, magicmenu, do_not_record, field, nogui, vfield, MagicTemplate, 
    set_design
)
from magicclass.widgets import Table, ToggleSwitch, Container, ComboBox, Separator
from magicclass.types import OneOf, Optional, Path, Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject, ComponentsViewer, ProjectSequence, CylindraCollectionProject
from cylindra.const import GlobalVariables as GVar, nm, MoleculesHeader as Mole

from ..widget_utils import FileFilter
from ..sta import StaParameters, INTERPOLATION_CHOICES, METHOD_CHOICES, _get_alignment

from .menus import File, SubtomogramAnalysis
from ._sequence import get_collection, ProjectSequenceEdit

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_ZRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_YRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_XRotation = Annotated[tuple[float, float], {"options": {"max": 90.0, "step": 0.1}}]
_MaxShifts = Annotated[tuple[nm, nm, nm], {"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"}]
_SubVolumeSize = Annotated[Optional[nm], {"text": "Use template shape", "options": {"value": 12., "max": 100.}, "label": "size (nm)"}]


@magicclass(
    widget_type="scrollable",
    properties={"min_height": 240},
    symbol=Expr("getattr", ["ui", "collection_analyzer"]),
)
class ProjectCollectionWidget(MagicTemplate):
    
    # Menus
    file = field(File)
    subtomogram_analysis = field(SubtomogramAnalysis)
    
    collection = ProjectSequenceEdit  # list of projects
    
    @magicclass(widget_type="collapsible", name="Subtomogram analysis")
    class StaWidget(MagicTemplate):
        params = StaParameters
    
    def __post_init__(self):
        self._project_sequence = ProjectSequence()
        self.StaWidget.params._get_scale = self._get_scale

    def _get_scale(self):
        return self._project_sequence._scale_validator.value

    @File.wraps
    @set_design(text="Load child projects")
    def add_children(self, paths: list[Path.Read[FileFilter.JSON]]):
        """Load a project json file."""
        for path in paths:
            self._project_sequence.add(path)
            self.collection.projects._add(path)
        return 

    @File.wraps
    @set_design(text="Load project")
    @do_not_record
    def load_project(self, path: Path.Read[FileFilter.JSON]):
        """Load a project json file."""
        project = CylindraCollectionProject.from_json(path)
        return project.to_gui(self)
    
    @File.wraps
    @set_design(text="Save project")
    def save_project(self, json_path: Path.Save[FileFilter.JSON]):
        """
        Save current project state as a json file.

        Parameters
        ----------
        json_path : Path
            Path of json file.
        """
        return CylindraCollectionProject.save_gui(self, Path(json_path))

    @collection.Buttons.wraps
    @set_design(text="Preview")
    @do_not_record
    def preview_all(self):
        """Preview project."""
        cbox = ComboBox(choices=self._get_project_paths)
        comp_viewer = ComponentsViewer()
        
        self.collection.changed.connect(lambda: cbox.reset_choices())
        cbox.changed.connect(
            lambda path: comp_viewer._from_project(CylindraProject.from_json(path))
        )
        cont = Container(widgets=[cbox, comp_viewer], labels=False)
        cont.native.setParent(self.native, cont.native.windowFlags())
        cont.show()
        cbox.changed.emit(cbox.value)

    @nogui
    @do_not_record
    def set_sequence(self, col: ProjectSequence):
        if not isinstance(col, ProjectSequence):
            raise TypeError(f"Expected a ProjectCollection, got {type(col)}")
        self._project_sequence = col
        for prj in col:
            self.collection.projects._add(prj.project_path)

    def _get_project_paths(self, w=None) -> list[Path]:
        return self.collection.projects.paths

    @subtomogram_analysis.wraps
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self, 
        paths: Bound[_get_project_paths],
        predicate: Bound[collection._get_expression],
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
    ):
        if size is None:
            shape = self.StaWidget.params._get_shape_in_nm()
        else:
            shape = (size, size, size)
        col = get_collection(paths, interpolation, shape, predicate=predicate)
        img = ip.asarray(col.average(), axes="zyx")
        img.set_scale(zyx=col.scale)
        
        return thread_worker.to_callback(
            self.StaWidget.params._show_reconstruction, img, f"[AVG]Collection"
        )
    
    @subtomogram_analysis.wraps
    @dask_thread_worker.with_progress(desc="Aligning all projects")
    def align_all(
        self, 
        paths: Bound[_get_project_paths], 
        predicate: Bound[collection._get_expression],
        template_path: Bound[StaWidget.params.template_path],
        mask_params: Bound[StaWidget.params._get_mask_params],
        tilt_range: Bound[StaWidget.params.tilt_range] = None,
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
    ):
        template = self.StaWidget.params._get_template(path=template_path)
        mask = self.StaWidget.params._get_mask(params=mask_params)
        shape = self.StaWidget.params._get_shape_in_nm()
        col = get_collection(paths, interpolation, shape, predicate=predicate)
        
        model_cls = _get_alignment(method)
        aligned = col.align(
            template=template.value, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=model_cls,
            tilt_range=tilt_range,
        )
        
        for _ids, mole in aligned.molecules.groupby(by=[Mole.id, "image-id"]):
            image_id: int = _ids[1]
            prj = self._project_sequence[image_id]
            mole.to_csv(prj.result_dir / f"aligned_{_ids[0]}.csv")
        return aligned
    