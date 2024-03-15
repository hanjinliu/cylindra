from typing import Annotated, Any, Literal

import polars as pl
from acryo import BatchLoader
from macrokit import Expr, Symbol
from magicclass import (
    MagicTemplate,
    confirm,
    do_not_record,
    field,
    get_button,
    magicclass,
    set_design,
)
from magicclass.types import Path
from magicclass.utils import thread_worker

from cylindra.const import FileFilter
from cylindra.core import ACTIVE_WIDGETS
from cylindra.project import CylindraBatchProject
from cylindra.widget_utils import POLARS_NAMESPACE, capitalize
from cylindra.widgets._accessors import BatchLoaderAccessor
from cylindra.widgets.batch._loaderlist import LoaderList
from cylindra.widgets.batch._sequence import PathInfo, ProjectSequenceEdit
from cylindra.widgets.batch._utils import LoaderInfo, TempFeatures
from cylindra.widgets.batch.sta import BatchSubtomogramAveraging


@magicclass(
    widget_type="split",
    layout="horizontal",
    name="Batch Analysis",
    properties={"min_height": 400},
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

    def _get_loader_paths(self, *_) -> list[PathInfo]:
        return [prj._get_loader_paths() for prj in self.constructor.projects]

    def _get_expression(self, *_):
        return self.constructor._get_expression()

    @set_design(text=capitalize, location=constructor)
    @thread_worker
    def construct_loader(
        self,
        paths: Annotated[Any, {"bind": _get_loader_paths}],
        predicate: Annotated[str | pl.Expr | None, {"bind": _get_expression}] = None,
        name: str = "Loader",
    ):  # fmt: skip
        """
        Construct a batch loader object from the given paths and predicate.

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
            for molecule_id, mole in enumerate(path_info.iter_molecules(_temp_feat)):
                loader.add_tomogram(img.value, mole, img_id)
                yield img_id / len(paths), molecule_id / len(path_info.molecules)
            yield (img_id + 1) / len(paths), 0.0
        yield 1.0, 1.0

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            loader = loader.filter(predicate)
        new = loader.replace(
            molecules=loader.molecules.drop_features(_temp_feat.to_drop),
            scale=self.constructor.scale.value,
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
        """
        Construct a batch loader from a list of project paths and a molecule pattern.

        Parameters
        ----------
        project_paths : list of path-like
            All the project paths to be used for construction.
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
        return None

    @set_design(text=capitalize, location=ProjectSequenceEdit.File)
    def construct_loader_by_pattern(
        self,
        path_pattern: str,
        mole_pattern: str = "*",
        predicate: Annotated[str | pl.Expr | None, {"bind": _get_expression}] = None,
        name: str = "Loader",
    ):
        """
        Construct a batch loader from a pattern of project paths and molecule paths.

        Parameters
        ----------
        path_pattern : str
            A glob pattern for project paths.
        mole_pattern : str, default "*"
            A glob pattern for molecule file names. For example, "*-ALN1.csv" will only
            collect the molecule file names ends with "-ALN1.csv".
        predicate : str or polars expression, optional
            Filter predicate of molecules.
        name : str, default "Loader"
            Name of the loader.
        """
        self.constructor.add_projects_glob(path_pattern, clear=True)
        self.constructor.select_molecules_by_pattern(mole_pattern)
        self.construct_loader(self._get_loader_paths(), predicate=predicate, name=name)
        return None

    def _add_loader(
        self,
        loader: BatchLoader,
        name: str,
        image_paths: dict[int, Path],
        invert: dict[int, bool],
    ):
        self._loaders.append(LoaderInfo(loader, name, image_paths, invert))
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
        return None

    @set_design(text=capitalize, location=ProjectSequenceEdit.MacroMenu)
    @do_not_record
    def show_native_macro(self):
        """Show the native macro widget of the batch analyzer."""
        self.macro.widget.show()
        ACTIVE_WIDGETS.add(self.macro.widget)
        return None

    @set_design(text="Load batch analysis project", location=ProjectSequenceEdit.File)
    @confirm(
        text="Are you sure to clear all loaders?", condition="len(self._loaders) > 0"
    )
    def load_batch_project(self, path: Path.Read[FileFilter.PROJECT]):
        """
        Load a batch project from a JSON file.

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
    def save_batch_project(
        self,
        save_path: Path.Save,
        molecules_ext: Literal[".csv", ".parquet"] = ".csv",
    ):
        """
        Save the GUI state to a JSON file.

        Parameters
        ----------
        save_path : path-like
            Path to the JSON file.
        molecules_ext : str, default ".csv"
            Extension of the molecule files.
        """
        return CylindraBatchProject.save_gui(self, Path(save_path), molecules_ext)
