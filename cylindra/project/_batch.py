import os
from typing import Union, TYPE_CHECKING
from pathlib import Path
import macrokit as mk
from pydantic import BaseModel

from cylindra.const import (
    get_versions,
    MoleculesHeader as Mole,
    nm,
)
from cylindra.project._base import BaseProject, PathLike, resolve_path, MissingWedge
from cylindra.project._utils import as_main_function
from cylindra._config import get_config

if TYPE_CHECKING:
    from cylindra.widgets.batch import CylindraBatchWidget


class ImageInfo(BaseModel):
    id: int
    image: PathLike
    scale: nm

    def resolve_path(self, file_dir: PathLike):
        self.image = resolve_path(self.image, Path(file_dir))
        return self


class LoaderInfoModel(BaseModel):
    molecule: PathLike
    images: list[ImageInfo]
    name: str
    scale: nm

    def resolve_path(self, file_dir: PathLike):
        file_dir = Path(file_dir)
        self.molecule = resolve_path(self.molecule, file_dir)
        self.images = [img.resolve_path(file_dir) for img in self.images]
        return self


class CylindraBatchProject(BaseProject):
    """A project of cylindra batch processing."""

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    loaders: list[LoaderInfoModel]
    template_image: Union[PathLike, None]
    mask_parameters: Union[None, tuple[float, float], PathLike]
    missing_wedge: MissingWedge = MissingWedge(params={}, kind="none")
    project_path: Union[Path, None] = None

    def _post_init(self):
        if hasattr(self, "tilt_range"):
            self.missing_wedge = MissingWedge.parse(self.tilt_range)
            del self.tilt_range

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.loaders = [ldr.resolve_path(file_dir) for ldr in self.loaders]
        self.template_image = resolve_path(self.template_image, file_dir)
        if isinstance(self.mask_parameters, (Path, str)):
            self.mask_parameters = resolve_path(self.mask_parameters, file_dir)
        return self

    @property
    def project_dir(self) -> Path:
        if self.project_path is None:
            raise ValueError("Project path is not set.")
        return self.project_path.parent

    @property
    def macro_path(self) -> Path:
        return self.project_dir / "script.py"

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraBatchWidget",
        json_path: Path,
        mole_ext: str = ".csv",
    ) -> "CylindraBatchProject":
        from datetime import datetime

        _versions = get_versions()
        results_dir = json_path.parent

        def as_relative(p: Path):
            assert isinstance(p, Path)
            try:
                out = p.relative_to(results_dir)
            except Exception:
                out = p
            return out

        loaders = list[LoaderInfoModel]()
        for info in gui._loaders:
            name = info.name
            loaders.append(
                LoaderInfoModel(
                    molecule=results_dir / f"Molecules-{name}{mole_ext}",
                    name=name,
                    images=[
                        ImageInfo(
                            id=id,
                            image=as_relative(fp),
                            scale=info.loader.scale,
                        )
                        for id, fp in info.image_paths.items()
                    ],
                    scale=info.loader.scale,
                )
            )

        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=next(iter(_versions.values())),
            dependency_versions=_versions,
            loaders=loaders,
            template_image=gui.sta.params.template_path,
            mask_parameters=gui.sta.params._get_mask_params(),
            missing_wedge=MissingWedge.parse(gui.sta.params.tilt_range),
            project_path=json_path,
        )

    @classmethod
    def save_gui(
        cls: "type[CylindraBatchProject]",
        gui: "CylindraBatchWidget",
        json_path: Path,
        mole_ext: str = ".csv",
    ) -> None:
        """Save the GUI state to a project directory."""
        results_dir = json_path.parent

        self = cls.from_gui(gui, json_path, mole_ext)

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # create a directory if not exists.

        # save molecules
        for lmodel, info in zip(self.loaders, gui._loaders):
            info.loader.molecules.to_file(lmodel.molecule)

        self.macro_path.write_text(as_main_function(gui.macro))

        # save objects
        self.to_json(json_path)
        return None

    def _to_gui(self, gui: "CylindraBatchWidget") -> None:
        import impy as ip
        from acryo import Molecules, BatchLoader

        for lmodel in self.loaders:
            loader = BatchLoader(scale=lmodel.scale)
            mole_dict = dict(Molecules.from_file(lmodel.molecule).groupby(Mole.image))
            for imginfo in lmodel.images:
                loader.add_tomogram(
                    image=ip.lazy.imread(imginfo.image, chunks=get_config().dask_chunk)
                    .set_scale(zyx=imginfo.scale)
                    .value,
                    molecules=mole_dict[imginfo.id],
                    image_id=imginfo.id,
                )
            image_paths = {imginfo.id: imginfo.image for imginfo in lmodel.images}
            gui._add_loader(loader, lmodel.name, image_paths)

        # append macro
        with open(self.macro_path) as f:
            txt = f.read()

        macro = mk.parse(txt)
        gui.macro.extend(macro.args)

        # load subtomogram analyzer state
        gui.sta.params.template_path = self.template_image or ""
        gui.sta.params._set_mask_params(self.mask_parameters)
        gui.reset_choices()
        return None
