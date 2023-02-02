from typing import Iterator, Union, TYPE_CHECKING, Annotated
from timeit import default_timer
from magicclass import (
    magicclass, magicmenu, do_not_record, field, vfield, MagicTemplate, 
    set_design, abstractapi, FieldGroup
)
from magicclass.widgets import HistoryFileEdit, Table, PushButton
from magicclass.types import OneOf, Optional, Path, Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker
from acryo import TomogramCollection, Molecules, SubtomogramLoader, alignment

import numpy as np
import impy as ip
import polars as pl
import napari

from cylindra.project import CylindraProject
from cylindra.collection import ProjectCollection
from cylindra.types import MonomerLayer
from cylindra import utils
from cylindra.const import (
    ALN_SUFFIX, MOLECULES, GlobalVariables as GVar, MoleculesHeader as Mole, nm
)

from .widget_utils import FileFilter
from .sta import StaParameters, INTERPOLATION_CHOICES, METHOD_CHOICES, _get_alignment
from . import widget_utils

if TYPE_CHECKING:
    from napari.layers import Image, Layer

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_ZRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_YRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_XRotation = Annotated[tuple[float, float], {"options": {"max": 90.0, "step": 0.1}}]
_MaxShifts = Annotated[tuple[nm, nm, nm], {"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"}]
_SubVolumeSize = Annotated[Optional[nm], {"text": "Use template shape", "options": {"value": 12., "max": 100.}, "label": "size (nm)"}]


class Project(FieldGroup):
    check = vfield(True, label="").with_options(text="")
    path = vfield("").with_options(enabled=False)
    preview_btn = vfield(widget_type=PushButton).with_options(text="Preview")
    
    @preview_btn.connect
    def _on_click(self):
        project = CylindraProject.from_json(self.path)
        mv = project.make_molecules_viewer()
        mv.native.setParent(self.native, mv.native.windowFlags())
        mv.show()
    
    @classmethod
    def from_path(cls, path: Path):
        self = cls(layout="horizontal", labels=False)
        self.path = str(path)
        self["path"].tooltip = self.path
        self.margins = (0, 0, 0, 0)
        return self

@magicclass(widget_type="scrollable", labels=False, record=False)
class ProjectPaths(MagicTemplate):
    def _add(self, path: Path):
        prj = Project.from_path(path)
        self.append(prj)
        self.min_height = min(len(self) * 30, 105)
    
    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def checked_indices(self) -> list[int]:
        indices: list[int] = []
        for i, wdt in enumerate(self):
            if wdt.check:
                indices.append(i)
        return indices
            
    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.path) for wdt in self]

@magicmenu
class SubtomogramAnalysis(MagicTemplate):
    """Analysis of subtomograms."""
    average_all = abstractapi()
    # calculate_fsc = abstractapi()

@magicmenu
class Refinement(MagicTemplate):
    """Refinement and alignment of subtomograms."""
    align_all = abstractapi()

@magicclass(widget_type="collapsible", name="Projects")
class CollectionProvider(MagicTemplate):
    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Buttons(MagicTemplate):
        check_all = abstractapi()
        add_projects = abstractapi()
        delete_projects = abstractapi()

    projects = field(ProjectPaths)
    filter_expression = vfield(str)

@magicclass(widget_type="scrollable")
class ProjectCollectionWidget(MagicTemplate):
    collection = CollectionProvider
    
    @magicclass(widget_type="collapsible", name="Subtomogram analysis")
    class StaWidget(MagicTemplate):
        subtomogram_analysis = field(SubtomogramAnalysis)
        refinement = field(Refinement)
        params = StaParameters
    
    def __post_init__(self):
        from qtpy import QtGui
        font = QtGui.QFont("Monospace", 10)
        font.setBold(True)
        self.collection["filter_expression"].native.setFont(font)
        self._project_collection = ProjectCollection()

    @collection.Buttons.wraps
    @set_design(text="✓", max_width=26)
    def check_all(self):
        """Check all projects."""
        for wdt in self.collection.projects:
            wdt.check = True
            
    @collection.Buttons.wraps
    @set_design(text="+", font_family="Arial")
    def add_projects(self, paths: Path.Multiple[FileFilter.JSON]):
        for path in paths:
            self._project_collection.add(path)
            self.collection.projects._add(path)

    @collection.Buttons.wraps    
    @set_design(text="-", font_family="Arial")
    def delete_projects(self):
        indices = self.collection.projects.checked_indices
        for idx in reversed(indices):
            del self.collection.projects[idx]
            del self._project_collection[idx]

    def preview_filtered_molecules(self):
        col = _get_collection(self.collection.projects.paths, predicate=self._get_expression())
        features = col.molecules.features
        if features.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = Table(value=features.to_pandas())
        table.read_only = True
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features", area="left")
        dock.setFloating(True)
        
    def _get_expression(self, w=None) -> pl.Expr:
        expr = self.collection.filter_expression
        if expr == "":
            return None
        return eval(expr, {"pl": pl}, {})
    
    
    def _get_project_paths(self, w=None) -> list[Path]:
        return self.collection.projects.paths

    @StaWidget.subtomogram_analysis.wraps
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self, 
        paths: Bound[_get_project_paths],
        predicate: Bound[_get_expression],
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
    ):
        if size is None:
            shape = self.StaWidget.params._get_shape_in_nm()
        else:
            shape = (size, size, size)
        col = _get_collection(paths, interpolation, shape, predicate=predicate)
        img = ip.asarray(col.average(), axes="zyx")
        img.set_scale(zyx=col.scale)
        
        return thread_worker.to_callback(
            self.StaWidget.params._show_reconstruction, img, f"[AVG]Collection"
        )
    
    @StaWidget.refinement.wraps
    @dask_thread_worker.with_progress(desc="Aligning all projects")
    def align_all(
        self, 
        paths: Bound[_get_project_paths], 
        predicate: Bound[_get_expression],
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
        col = _get_collection(paths, interpolation, shape, predicate=predicate)
        
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
        
        # TODO: how to save the results?
        return aligned
    
def _get_collection(
    project_paths: list[Path], 
    order: int = 3, 
    output_shape: tuple[nm, nm, nm] = None,
    predicate: "str | pl.Expr | None" = None,
):
    # check scales
    projects: list[CylindraProject] = []
    scales: list[nm] = []
    for path in project_paths:
        project = CylindraProject.from_json(path)
        projects.append(project)
        scales.append(project.scale)
    scale = scales[0]
    if len(set(scales)) > 1:
        raise ValueError("All projects must have the same scale.")

    if output_shape is None:
        col = TomogramCollection(order=order)
    else:
        output_shape = tuple(np.round(np.array(output_shape) / scale).astype(int))
        col = TomogramCollection(order=order, output_shape=output_shape, scale=scale)
        
    for project in projects:
        tomogram = ip.lazy_imread(project.image, chunks=GVar.daskChunk)

        for fp in project.molecules:
            fp = Path(fp)
            mole = Molecules.from_csv(fp)
            mole.features = mole.features.with_columns(pl.Series("file-name", np.full(len(mole), fp.stem)))
            col.add_tomogram(tomogram.value, molecules=mole)
    if predicate is not None:
        col = col.filter(predicate)
    return col
