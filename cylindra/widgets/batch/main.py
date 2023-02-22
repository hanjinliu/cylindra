from typing import Annotated, TYPE_CHECKING
from acryo import BatchLoader, Molecules
from macrokit import Symbol, Expr
from magicclass import (
    magicclass, do_not_record, field, nogui, MagicTemplate, set_design, set_options
)
from magicclass.types import OneOf, Optional, Path, Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker

import numpy as np
import impy as ip

from cylindra.project import ProjectSequence, CylindraBatchProject
from cylindra.const import nm, MoleculesHeader as Mole, GlobalVariables as GVar
from cylindra.utils import roundint

from ..widget_utils import FileFilter
from .. import widget_utils
from ..sta import StaParameters, INTERPOLATION_CHOICES, METHOD_CHOICES, _get_alignment

from .menus import Projects, SubtomogramAnalysis, Macro
from ._sequence import ProjectSequenceEdit

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
class CylindraBatchWidget(MagicTemplate):
    
    # Menus
    Projects = field(Projects)
    SubtomogramAnalysis = field(SubtomogramAnalysis, name="Subtomogram analysis")
    MacroMenu = field(Macro, name="Macro")

    _loader_constructor = ProjectSequenceEdit
    
    @magicclass(widget_type="collapsible", name="Subtomogram analysis")
    class StaWidget(MagicTemplate):
        params = StaParameters
    
    def __post_init__(self):
        self._current_loader: "BatchLoader | None" = None

    @_loader_constructor.wraps
    def construct_loader(
        self,
        paths: Bound[_loader_constructor._get_loader_paths],
        predicate: Bound[_loader_constructor._get_expression],
    ):
        loader = BatchLoader()
        scales: list[float] = []
        for img_id, (img_path, mole_paths) in enumerate(paths):
            img = ip.lazy_imread(img_path, chunks=GVar.daskChunk)
            scales.append(img.scale.x)
            for mole_path in mole_paths:
                mole = Molecules.from_csv(mole_path)
                loader.add_tomogram(img.value, mole, img_id)
            
        if abs(max(scales) / min(scales) - 1) > 0.01:
            raise ValueError("Scale error must be less than 1%.")
        if predicate is not None:
            loader = loader.filter(predicate)
        self._current_loader = loader.replace(scale=np.mean(scales))
        return self._current_loader


    @property
    def current_loader(self) -> BatchLoader:
        if self._current_loader is None:
            raise ValueError("No loader exists.")
        return self._current_loader
    
    @property
    def template(self) -> "ip.ImgArray | None":
        """Template image."""
        loader = self.current_loader
        template, _ = loader.normalize_input(self.StaWidget.params._get_template())
        return ip.asarray(template, axes="zyx").set_scale(zyx=loader.scale)
    
    @property
    def mask(self) -> "ip.ImgArray | None":
        """Mask image."""
        loader = self.current_loader
        _, mask = loader.normalize_input(
            self.StaWidget.params._get_template(allow_none=True), 
            self.StaWidget.params._get_mask()
        )
        return ip.asarray(mask, axes="zyx").set_scale(zyx=loader.scale)

    @Projects.wraps
    @set_design(text="Load")
    def open_constructor(self):
        self._loader_constructor.show()
    
    @Projects.wraps
    @set_design(text="Load batch project")
    @do_not_record
    def load_project(self, path: Path.Read[FileFilter.JSON]):
        """Load a project json file."""
        project = CylindraBatchProject.from_json(path)
        return project.to_gui(self)
    
    @Projects.wraps
    @set_design(text="Save batch project")
    def save_project(self, json_path: Path.Save[FileFilter.JSON]):
        """
        Save current project state as a json file.

        Parameters
        ----------
        json_path : Path
            Path of json file.
        """
        return CylindraBatchProject.save_gui(self, Path(json_path))


    @nogui
    @do_not_record
    def set_sequence(self, col: ProjectSequence):
        if not isinstance(col, ProjectSequence):
            raise TypeError(f"Expected a ProjectCollection, got {type(col)}")
        for prj in col:
            self._loader_constructor.projects._add(prj.project_path)
        self.reset_choices()
    
    @nogui
    @do_not_record
    def get_loader(self, order: int = 3, output_shape=None, predicate=None) -> BatchLoader:
        return self._loader_constructor._get_batch_loader(order=order, output_shape=output_shape, predicate=predicate)

    @SubtomogramAnalysis.wraps
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self, 
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
    ):
        if size is None:
            shape = self.template.shape
        else:
            shape = self._get_shape_in_px(roundint(size / self.current_loader.scale))
        loader = self.current_loader.replace(output_shape=shape, order=interpolation)
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale)
        
        return thread_worker.to_callback(
            self.StaWidget.params._show_reconstruction, img, f"[AVG]Collection"
        )
    
    @SubtomogramAnalysis.wraps
    @dask_thread_worker.with_progress(desc="Aligning all projects")
    def align_all(
        self,
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
        
        template, mask = self.current_loader.normalize_input(
            template=self.StaWidget.params._get_template(path=template_path),
            mask=self.StaWidget.params._get_mask(params=mask_params),
        )
        loader = self.current_loader.replace(output_shape=template.shape, order=interpolation)
        aligned = loader.align(
            template=template,
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model= _get_alignment(method),
            tilt_range=tilt_range,
        )
        
        self._current_loader = aligned
        # for _ids, mole in aligned.molecules.groupby(by=[Mole.id, Mole.image]):
        #     image_id: int = _ids[1]
        #     prj = self._project_sequence[image_id]
        #     mole.to_csv(prj.result_dir / f"aligned_{_ids[0]}.csv")
        return aligned

    @SubtomogramAnalysis.wraps
    @set_design(text="Calculate FSC")
    @dask_thread_worker.with_progress(desc="Calculating FSC")
    def calculate_fsc(
        self,
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
        mask = self.mask
        shape = self._get_shape_in_px(roundint(size / self.current_loader.scale))
        loader = self.current_loader.replace(output_shape=shape, order=interpolation)
    
        if mask is None:
            mask = 1.
        if dfreq is None:
            dfreq = 1.5 / min(shape) * loader.scale
        img = ip.asarray(
            loader.average_split(n_set=n_set, seed=seed, squeeze=False),
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
            img_avg = ip.asarray(img[0, 0] + img[0, 1], axes="zyx") / img.shape.i
            img_avg.set_scale(zyx=loader.scale)
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
                widget_utils.plot_fsc(freq, fsc_mean, fsc_std, [crit_0143, crit_0500], loader.scale)
            
            resolution_0143 = widget_utils.calc_resolution(freq, fsc_mean, crit_0143, loader.scale)
            resolution_0500 = widget_utils.calc_resolution(freq, fsc_mean, crit_0500, loader.scale)
            
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
    
    @MacroMenu.wraps
    @do_not_record
    def show_macro(self):
        from cylindra import instance
        ui = instance()
        macro_str = self.macro.widget.textedit.value
        win = ui.macro.widget.new_window("Batch")
        win.textedit.value = macro_str
        win.show()
        return None
    
    @MacroMenu.wraps
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        return None
    
    def _get_shape_in_px(self, default: int = None) -> tuple[int, ...]:
        if default is None:
            tmp = self.template
            return tuple(np.array(tmp.shape))
        else:
            return (default,) * 3
