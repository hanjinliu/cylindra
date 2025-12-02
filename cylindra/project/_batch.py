import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import macrokit as mk
from pydantic import BaseModel, Field

from cylindra._config import get_config
from cylindra._io import lazy_imread
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import get_versions, nm
from cylindra.project._base import BaseProject, PathLike, resolve_path
from cylindra.project._utils import as_main_function, resolve_relative_paths

if TYPE_CHECKING:
    from cylindra.widgets.batch import CylindraBatchWidget


class ImageInfo(BaseModel):
    """Model that describe how to load an image."""

    id: int
    image: PathLike
    scale: nm
    invert: bool = False

    def resolve_path(self, file_dir: PathLike):
        self.image = resolve_path(self.image, Path(file_dir))
        return self


class LoaderInfoModel(BaseModel):
    """Model that describe how to construct a batch loader."""

    molecule: PathLike
    images: list[ImageInfo]
    name: str
    scale: nm

    def resolve_path(self, file_dir: PathLike):
        file_dir = Path(file_dir)
        self.molecule = resolve_path(self.molecule, file_dir)
        self.images = [img.resolve_path(file_dir) for img in self.images]
        return self


class ChildProjectInfo(BaseModel):
    """Model that describes the state of a child project."""

    path: Path
    spline_selected: list[bool] = Field(default_factory=list)
    molecules_selected: list[bool] = Field(default_factory=list)


class CylindraBatchProject(BaseProject):
    """A project of cylindra batch processing."""

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    children: list[ChildProjectInfo] = []
    loaders: list[LoaderInfoModel]
    project_path: Path | None = None

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.loaders = [ldr.resolve_path(file_dir) for ldr in self.loaders]
        return self

    @property
    def macro_path(self) -> Path:
        return self.project_path / "script.py"

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraBatchWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
    ) -> "CylindraBatchProject":
        _versions = get_versions()

        def as_relative(p: Path):
            assert isinstance(p, Path)
            try:
                out = p.relative_to(project_dir)
            except Exception:
                out = p
            return out

        loaders = list[LoaderInfoModel]()
        for info in gui._loaders:
            name = info.name
            loaders.append(
                LoaderInfoModel(
                    molecule=project_dir / f"Molecules-{name}{mole_ext}",
                    name=name,
                    images=[
                        ImageInfo(
                            id=id,
                            image=as_relative(fp),
                            scale=info.loader.scale,
                            invert=info.invert.get(id, False),
                        )
                        for id, fp in info.image_paths.items()
                    ],
                    scale=info.loader.scale,
                )
            )

        children = list[ChildProjectInfo]()
        for child_widget in gui.constructor.projects:
            child_project_path = as_relative(Path(child_widget.path))
            if child_project_path is None:
                warnings.warn(
                    f"Child project path {child_widget.path!r} could not be resolved "
                    f"as relative to {project_dir}, skipping.",
                    UserWarning,
                    stacklevel=1,
                )
                continue
            spl_checked = [s.check for s in child_widget.splines]
            mol_checked = [m.check for m in child_widget.molecules]
            info = ChildProjectInfo(
                path=child_project_path,
                spline_selected=spl_checked,
                molecules_selected=mol_checked,
            )
            children.append(info)
        return cls.from_children(
            children=children, loaders=loaders, project_dir=project_dir
        )

    @classmethod
    def from_children(
        cls,
        children: list[ChildProjectInfo],
        loaders: list[LoaderInfoModel] = [],
        project_dir: Path | None = None,
    ) -> "CylindraBatchProject":
        from datetime import datetime

        _versions = get_versions()
        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=next(iter(_versions.values())),
            dependency_versions=_versions,
            children=children,
            loaders=list(loaders),
            project_path=project_dir,
        )

    def save(self, project_dir: Path) -> None:
        """Save the project to a directory."""
        if not project_dir.exists():
            project_dir.mkdir()

        project_dir.joinpath("script.py").write_text(
            as_main_function(mk.parse("", squeeze=False))
        )

        # save objects
        self.to_json(project_dir / "project.json")

    @classmethod
    def save_gui(
        cls: "type[CylindraBatchProject]",
        gui: "CylindraBatchWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
    ) -> None:
        """Save the GUI state to a project directory."""
        self = cls.from_gui(gui, project_dir, mole_ext)

        if not project_dir.exists():
            project_dir.mkdir()

        # save molecules
        for lmodel, info in zip(self.loaders, gui._loaders, strict=True):
            info.loader.molecules.to_file(lmodel.molecule)

        self.project_path.joinpath("script.py").write_text(as_main_function(gui.macro))

        # save objects
        self.to_json(project_dir / "project.json")

    def _to_gui(self, gui: "CylindraBatchWidget") -> None:
        from acryo import BatchLoader, Molecules

        gui.constructor.clear_projects()
        for child in self.children:
            # solve relative path
            child_path = resolve_relative_paths(child.path, self.project_path)
            if child_path is None:
                warnings.warn(
                    f"Child project path {child.path!r} could not be resolved "
                    f"as relative to {self.project_path}, skipping.",
                    UserWarning,
                    stacklevel=1,
                )
                continue
            gui.constructor.add_projects([child_path], clear=False)
            last_project = gui.constructor.projects[-1]
            for spl_widget, checked in zip(
                iter(last_project.splines), child.spline_selected, strict=False
            ):
                spl_widget.check = checked
            for mol_widget, checked in zip(
                iter(last_project.molecules), child.molecules_selected, strict=False
            ):
                mol_widget.check = checked
        for lmodel in self.loaders:
            loader = BatchLoader(scale=lmodel.scale)
            mole_dict = dict(Molecules.from_file(lmodel.molecule).groupby(Mole.image))
            for imginfo in lmodel.images:
                loader.add_tomogram(
                    image=lazy_imread(imginfo.image, chunks=get_config().dask_chunk)
                    .set_scale(zyx=imginfo.scale)
                    .value,
                    molecules=mole_dict[imginfo.id],
                    image_id=imginfo.id,
                )
            image_paths = {imginfo.id: imginfo.image for imginfo in lmodel.images}
            invert = {imginfo.id: imginfo.invert for imginfo in lmodel.images}
            gui._add_loader(loader, lmodel.name, image_paths, invert)

        txt = self.project_path.joinpath("script.py").read_text()
        macro = mk.parse(txt)
        gui.macro.extend(macro.args)
        gui.reset_choices()
