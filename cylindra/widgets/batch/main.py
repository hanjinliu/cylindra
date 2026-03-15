from typing import Annotated, Any, Iterator, Literal

import polars as pl
from acryo import BatchLoader, Molecules
from macrokit import Expr, Symbol
from magicclass import (
    MagicTemplate,
    confirm,
    do_not_record,
    field,
    get_button,
    magicclass,
    nogui,
    set_design,
)
from magicclass.types import Optional, Path
from magicclass.utils import thread_worker
from qtpy.QtWidgets import QSizePolicy

from cylindra.components import CylSpline
from cylindra.const import FileFilter
from cylindra.core import ACTIVE_WIDGETS
from cylindra.project import CylindraBatchProject, CylindraProject
from cylindra.utils import parse_tilt_model, unwrap_wildcard
from cylindra.widget_utils import POLARS_NAMESPACE, capitalize
from cylindra.widgets._accessors import BatchLoaderAccessor
from cylindra.widgets.batch._loaderlist import LoaderList
from cylindra.widgets.batch._sequence import PathInfo, ProjectSequenceEdit
from cylindra.widgets.batch._utils import LoaderInfo, TempFeatures
from cylindra.widgets.batch.sta import BatchSubtomogramAveraging
from cylindra.widgets.subwidgets.misc import TiltModelEdit


@magicclass(
    widget_type="split",
    layout="horizontal",
    name="Batch Analysis",
    properties={"min_height": 360},
    symbol=Expr("getattr", [Symbol("ui"), "batch"]),
)
class CylindraBatchWidget(MagicTemplate):
    constructor = field(ProjectSequenceEdit)
    sta = field(BatchSubtomogramAveraging)
    loader_infos = BatchLoaderAccessor()

    def __init__(self):
        self._loaders = LoaderList()
        self._loaders.events.inserted.connect(self.reset_choices)
        self._loaders.events.removed.connect(self.reset_choices)
        self._loaders.events.moved.connect(self.reset_choices)

    def __post_init__(self):
        self.native.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.sta.visible = False

    def _get_loader_paths(self, *_) -> list[PathInfo]:
        return [prj._get_loader_paths() for prj in self.constructor.projects]

    def _get_expression(self, *_):
        return self.constructor._get_expression()

    def _get_constructor_scale(self, *_) -> float:
        return self.constructor.scale.value

    @set_design(text=capitalize, location=ProjectSequenceEdit.File)
    @do_not_record
    def new_projects(
        self,
        paths: Path.Multiple[FileFilter.IMAGE],
        save_root: Path.Save[FileFilter.DIRECTORY],
        ref_paths: Path.Multiple[FileFilter.IMAGE] = [],
        scale: Annotated[Optional[float], {"text": "Use image original scale", "options": {"min": 0.01, "step": 0.0001},}] = None,
        tilt_model: Annotated[dict, {"widget_type": TiltModelEdit}] = None,
        bin_size: list[int] = [1],
        invert: bool = False,
        extension: Literal["", ".zip", ".tar"] = "",
        strip_prefix: str = "",
        strip_suffix: str = "",
        overwrite: bool = True,
    ):  # fmt: skip
        """Create new projects from images.

        This method is usually used for batch processing, efficient visual inspection,
        and particle picking.

        Parameters
        ----------
        paths : list of str or Path
            A list of image paths or wildcard patterns, such as "path/to/*.mrc".
        save_root : str or Path
            The root directory to save the output projects.
        ref_paths : list of str or Path, optional
            A list of reference image paths. Reference images are usually a binned or
            denoised version of the original images.
        scale : float, optional
            The scale of the images in nanometers. If None, the original scale of the
            images will be used.
        tilt_model : dict, optional
            A tilt model that describes the tilt angles and axis.
        bin_size : list of int, default [1]
            Initial bin size of image. Binned image will be used for visualization in
            the viewer. You can use both binned and non-binned image for analysis.
        invert : bool, default False
            Whether to invert the image intensities.
        extension : str, default ""
            The file extension (or directory) of the saved project files.
        strip_prefix : str, default ""
            A prefix to strip from the project name.
        strip_suffix : str, default ""
            A suffix to strip from the project name.
        overwrite : bool, default True
            If child project files of the same name already exist under the save root,
            they will be overwritten. This is useful when cylindra batch project is
            imported from file outputs of a long-running job from other softwares.
        """
        _paths = unwrap_wildcard(paths)
        self._new_projects_from_table(
            _paths,
            save_root=save_root,
            ref_paths=unwrap_wildcard(ref_paths) or None,
            scale=[scale] * len(_paths),
            tilt_model=[tilt_model] * len(_paths),
            bin_size=[bin_size] * len(_paths),
            invert=[invert] * len(_paths),
            extension=extension,
            strip_prefix=strip_prefix,
            strip_suffix=strip_suffix,
            overwrite=overwrite,
        )

    def _new_projects_from_table(
        self,
        path: list[Path],
        save_root: Path,
        ref_paths: list[Path] | None = None,
        scale: list[float | None] | None = None,
        tilt_model: list[dict | None] | None = None,
        bin_size: list[list[int]] | None = None,
        invert: list[bool] | None = None,
        splines: list["CylSpline"] | None = None,
        molecules: list[dict[str, Molecules]] | None = None,
        extension: Literal["", ".zip", ".tar"] = "",
        strip_prefix: str = "",
        strip_suffix: str = "",
        overwrite: bool = True,
    ):
        projects = list[tuple[CylindraProject, str]]()
        num_projects = len(path)
        for img_path, _scale, _ref, tlt, _bin_size, _inv in zip(
            path,
            _or_default_list(scale, None, num_projects),
            _or_default_list(ref_paths, None, num_projects),
            _or_default_list(tilt_model, None, num_projects),
            _or_default_list(bin_size, [1], num_projects),
            _or_default_list(invert, False, num_projects),
            strict=True,
        ):
            each_project = CylindraProject.new(
                img_path,
                scale=_scale,
                image_reference=_ref,
                multiscales=_bin_size,
                missing_wedge=tlt,
                invert=_inv,
            )
            prj_name = img_path.stem
            projects.append((each_project, prj_name))
        if len(projects) == 0:
            raise ValueError("No projects created.")
        save_root.mkdir(parents=True, exist_ok=True)
        self.constructor.projects.clear()
        for (prj, prj_name), spl, mole in zip(
            projects,
            splines or [[]] * num_projects,
            molecules or [{}] * num_projects,
            strict=True,
        ):
            if strip_prefix and prj_name.startswith(strip_prefix):
                prj_name = prj_name[len(strip_prefix) :]
            if strip_suffix and prj_name.endswith(strip_suffix):
                prj_name = prj_name[: -len(strip_suffix)]
            save_path = save_root / f"{prj_name}{extension}"
            if save_path.exists() and not overwrite:
                prj = CylindraProject.from_file(save_path)
            else:
                prj.save(save_path, splines=spl, molecules=mole)
                prj.project_path = save_path
            self.constructor.projects._add(prj.project_path)
        self.save_batch_project(save_path=save_root)
        self.show()

    @set_design(text=capitalize, location=constructor)
    @thread_worker
    def construct_loader(
        self,
        paths: Annotated[Any, {"bind": _get_loader_paths}],
        predicate: Annotated[str | pl.Expr | None, {"bind": _get_expression}] = None,
        name: str = "Loader",
        scale: Annotated[float | None, {"bind": _get_constructor_scale}] = None,
    ):  # fmt: skip
        """Construct a batch loader object from the given paths and predicate.

        Parameters
        ----------
        paths : list of (Path, list[Path]) or list of (Path, list[Path], Path)
            List of tuples of image path, list of molecule paths, and project path. The
            project path is optional.
        predicate : str or polars expression, optional
            Filter predicate of molecules.
        name : str, default "Loader"
            Name of the loader.
        """
        if name == "":
            raise ValueError("Name must be given.")

        yield 0.0, 0.0  # this function yields the progress
        loader = BatchLoader()
        image_paths = dict[int, Path]()
        invert = dict[int, bool]()
        _temp_feat = TempFeatures()
        for img_id, path_info in enumerate(paths):
            path_info = PathInfo(*path_info)
            img = path_info.lazy_imread()
            image_paths[img_id] = Path(path_info.image)
            invert[img_id] = path_info.need_invert
            if scale is None:
                if prj := path_info.project_instance():
                    scale = prj.scale
                else:
                    scale = img.scale.x
            if prj := path_info.project_instance():
                tilt = prj.missing_wedge.as_param()
                if tilt is not None:
                    tilt = parse_tilt_model(tilt)
            else:
                tilt = None
            for molecule_id, mole in enumerate(
                path_info.iter_molecules(_temp_feat, scale)
            ):
                loader.add_tomogram(img.value, mole, img_id, tilt_model=tilt)
                yield img_id / len(paths), molecule_id / len(path_info.molecules)
            yield (img_id + 1) / len(paths), 0.0
        yield 1.0, 1.0

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            loader = loader.filter(predicate)
        new = loader.replace(
            molecules=loader.molecules.drop_features(_temp_feat.to_drop),
            scale=scale,
        )

        @thread_worker.callback
        def _on_return():
            self._add_loader(new, name, image_paths, invert)

        return _on_return

    @construct_loader.yielded.connect
    def _on_construct_loader_yielded(self, prog: tuple[float, float]):
        btn = get_button(self.construct_loader, cache=True)
        btn.text = f"Constructing... ({prog[0]:.1%}, {prog[1]:.1%})"

    @construct_loader.finished.connect
    def _on_construct_loader_finished(self):
        btn = get_button(self.construct_loader, cache=True)
        btn.text = "Construct loader"

    @set_design(text=capitalize, location=ProjectSequenceEdit.File)
    def construct_loader_by_list(
        self,
        project_paths: Path.Multiple[FileFilter.PROJECT],
        mole_pattern: str = "*",
        predicate: Annotated[str | pl.Expr | None, {"bind": _get_expression}] = None,
        name: str = "Loader",
    ):
        """Construct a batch loader from a list of project paths and a molecule pattern.

        Parameters
        ----------
        project_paths : list of path-like
            All the project paths to be used for construction. Entries can contain
            glob patterns such as "*" and "?".
        mole_pattern : str, default "*"
            A glob pattern for molecule file names. For example, "*-ALN1.csv" will only
            collect the molecule file names ends with "-ALN1.csv".
        predicate : str or polars expression, optional
            Filter predicate of molecules.
        name : str, default "Loader"
            Name of the loader.
        """
        self.constructor.add_projects(project_paths, clear=True)
        self.constructor.select_molecules_by_pattern(mole_pattern)
        self.construct_loader(self._get_loader_paths(), predicate=predicate, name=name)

    def _add_loader(
        self,
        loader: BatchLoader,
        name: str,
        image_paths: dict[int, Path],
        invert: dict[int, bool],
    ):
        self._loaders.append(LoaderInfo(loader, name, image_paths, invert))
        self.sta.visible = True
        try:
            self.sta["loader_name"].value = self.sta["loader_name"].choices[-1]
        except Exception:
            pass  # Updating the value is not important. Silence just in case.

    @set_design(text=capitalize, location=ProjectSequenceEdit.MacroMenu)
    @do_not_record
    def show_macro(self):
        """Show the macro widget of the batch analyzer."""
        from cylindra import instance

        ui = instance()
        assert ui is not None
        macro_str = self.macro.widget.textedit.value
        ui.OthersMenu.Macro._get_macro_window(macro_str, "Batch")

    @set_design(text=capitalize, location=ProjectSequenceEdit.MacroMenu)
    @do_not_record
    def show_native_macro(self):
        """Show the native macro widget of the batch analyzer."""
        self.macro.widget.show()
        ACTIVE_WIDGETS.add(self.macro.widget)

    @set_design(text="Load batch analysis project", location=ProjectSequenceEdit.File)
    @confirm(
        text="Are you sure to clear all loaders?", condition="len(self._loaders) > 0"
    )
    def load_batch_project(self, path: Path.Read[FileFilter.PROJECT]):
        """Load a batch project from a JSON file.

        Parameters
        ----------
        path : path-like
            Path to the JSON file.
        """
        self._loaders.clear()
        return CylindraBatchProject.from_file(path)._to_gui(self)

    @set_design(
        text="Save as batch analysis project", location=ProjectSequenceEdit.File
    )
    @do_not_record
    def save_batch_project(
        self,
        save_path: Path.Save,
        molecules_ext: Literal[".csv", ".parquet"] = ".csv",
    ):
        """Save the GUI state to a JSON file.

        Parameters
        ----------
        save_path : path-like
            Path to the JSON file.
        molecules_ext : str, default ".csv"
            Extension of the molecule files.
        """
        return CylindraBatchProject.save_gui(self, Path(save_path), molecules_ext)

    @nogui
    def iter_projects(self) -> Iterator[CylindraProject]:
        """Iterate over all projects in the batch project."""
        for prj_widget in self.constructor.projects:
            yield prj_widget.project


def _or_default_list(value, default, length: int):
    """Return value if it is a list, otherwise return default."""
    if value is None:
        return [default] * length
    if len(value) != length:
        raise ValueError(f"Expected {length} items, got {len(value)}")
    return value
