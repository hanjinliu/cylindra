from typing import Annotated, TYPE_CHECKING
from macrokit import Symbol, Expr
from magicclass import (
    magicclass, do_not_record, field, nogui, MagicTemplate, set_design
)
from magicclass.types import OneOf, Optional, Path, Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker

import numpy as np
import impy as ip

from cylindra.project import ProjectSequence, CylindraCollectionProject
from cylindra.const import nm, MoleculesHeader as Mole

from ..widget_utils import FileFilter
from .. import widget_utils
from ..sta import StaParameters, INTERPOLATION_CHOICES, METHOD_CHOICES, _get_alignment

from .menus import File, SubtomogramAnalysis, Macro
from ._sequence import get_collection, ProjectSequenceEdit

if TYPE_CHECKING:
    from napari.layers import Image

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
    symbol=Expr("getattr", [Symbol("ui"), "batch"]),
)
class ProjectCollectionWidget(MagicTemplate):
    
    # Menus
    file = field(File, name="File")
    subtomogram_analysis = field(SubtomogramAnalysis, name="Subtomogram analysis")
    
    # MacroMenu = field(Macro, name="Macro")
    
    collection = ProjectSequenceEdit  # list of projects
    
    @magicclass(widget_type="collapsible", name="Subtomogram analysis")
    class StaWidget(MagicTemplate):
        params = StaParameters
    
    def __post_init__(self):
        self._project_sequence = ProjectSequence()
        self.StaWidget.params._get_scale = self._get_scale

    @property
    def project_sequence(self) -> ProjectSequence:
        return self.collection.projects._project_sequence

    def _get_scale(self):
        return self.project_sequence._scale_validator.value

    @File.wraps
    @set_design(text="Add projects")
    def add_children(self, paths: list[Path.Read[FileFilter.JSON]]):
        """Add project json files as the child projects."""
        for path in paths:
            self.project_sequence.add(path)
            self.collection.projects._add(path)
        return 
    
    @File.wraps
    @set_design(text="Add projects with wildcard path")
    def add_children_glob(self, pattern: str):
        """Add project json files using wildcard path."""
        import glob

        pattern = str(pattern)
        for path in glob.glob(pattern):
            self.project_sequence.add(path)
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

    @nogui
    @do_not_record
    def set_sequence(self, col: ProjectSequence):
        if not isinstance(col, ProjectSequence):
            raise TypeError(f"Expected a ProjectCollection, got {type(col)}")
        self.collection.projects._project_sequence = col
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
        shape = self._get_shape_in_nm(size)
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
        
        for _ids, mole in aligned.molecules.groupby(by=[Mole.id, Mole.image]):
            image_id: int = _ids[1]
            prj = self._project_sequence[image_id]
            mole.to_csv(prj.result_dir / f"aligned_{_ids[0]}.csv")
        return aligned

    @subtomogram_analysis.wraps
    @set_design(text="Calculate FSC")
    @dask_thread_worker.with_progress(desc="Calculating FSC")
    def calculate_fsc(
        self,
        paths: Bound[_get_project_paths], 
        predicate: Bound[collection._get_expression],
        mask_params: Bound[StaWidget.params._get_mask_params],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: Annotated[Optional[float], {"label": "Frequency precision", "text": "Choose proper value", "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02}}] = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        {layer}{mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_set : int, default is 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default is True
            If true, subtomogram averaging will be shown after FSC calculation.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        mask = self.StaWidget.params._get_mask(params=mask_params)
        shape = self._get_shape_in_nm(size)
        col = get_collection(paths, interpolation, shape, predicate=predicate)
    
        if mask is None:
            mask = 1.
        if dfreq is None:
            dfreq = 1.5 / min(shape) * col.scale
        img = ip.asarray(
            col.average_split(n_set=n_set, seed=seed, squeeze=False),
            axes="ipzyx",
        )
        
        # NOTE: images added with a big constant offset cause strong correlation at the
        # masked edges. Here to avoid it, normalize images to minimize the artifact.
        img -= img.mean()
        img: ip.ImgArray

        fsc_all: list[np.ndarray] = []
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = ip.fsc(img0 * mask, img1 * mask, dfreq=dfreq)
            fsc_all.append(fsc)
        if show_average:
            img_avg = ip.asarray(img[0, 0] + img[0, 1], axes="zyx") / len(img.shape.i)
            img_avg.set_scale(zyx=col.scale)
        else:
            img_avg = None

        fsc_all = np.stack(fsc_all, axis=1)
        
        @thread_worker.to_callback
        def _calculate_fsc_on_return():
            from cylindra import instance
            fsc_mean = np.mean(fsc_all, axis=1)
            fsc_std = np.std(fsc_all, axis=1)
            crit_0143 = 0.143
            crit_0500 = 0.500
            
            parent = instance()
            parent.log.print_html("<b>Fourier Shell Correlation of the collection</b>")
            with parent.log.set_plt(rc_context={"font.size": 15}):
                widget_utils.plot_fsc(freq, fsc_mean, fsc_std, [crit_0143, crit_0500], col.scale)
            
            resolution_0143 = widget_utils.calc_resolution(freq, fsc_mean, crit_0143, col.scale)
            resolution_0500 = widget_utils.calc_resolution(freq, fsc_mean, crit_0500, col.scale)
            
            parent.log.print_html(f"Resolution at FSC=0.5 ... <b>{resolution_0500:.3f} nm</b>")
            parent.log.print_html(f"Resolution at FSC=0.143 ... <b>{resolution_0143:.3f} nm</b>")
            parent._LoggerWindow.show()
            
            if img_avg is not None:
                _rec_layer: "Image" = self.StaWidget.params._show_reconstruction(
                    img_avg, name = "[AVG]Collection",
                )
                _rec_layer.metadata["fsc"] = widget_utils.FscResult(
                    freq, fsc_mean, fsc_std, resolution_0143, resolution_0500
                )
        return _calculate_fsc_on_return
    
    def _get_shape_in_nm(self, default: int = None) -> tuple[int, ...]:
        if default is None:
            return self.StaWidget.params._get_shape_in_nm()
        else:
            return (default,) * 3
