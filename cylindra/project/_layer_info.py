import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acryo import Molecules
    from napari.layers import Layer
    from pydantic import BaseModel

    from cylindra._napari import LandscapeSurface, MoleculesLayer
    from cylindra.widgets.main import CylindraMainWidget
else:
    from pydantic_compat import BaseModel


class LayerInfo(BaseModel):
    @classmethod
    @abstractmethod
    def from_layer(cls, gui: "CylindraMainWidget", layer: "Layer") -> "LayerInfo":
        """Convert layer to info."""

    @abstractmethod
    def to_layer(self, gui: "CylindraMainWidget", project_dir: Path) -> "Layer | None":
        """Convert info to layer."""

    @abstractmethod
    def save_layer(self, gui: "CylindraMainWidget", dir: Path):
        """Save layer to project directory."""


class MoleculeColormap(BaseModel):
    cmap: list[tuple[float, str]]
    limits: tuple[float, float]
    by: str


class MoleculesInfo(LayerInfo):
    """Info of molecules layer."""

    name: str = "#unknown"  # including extension
    source: int | None = None
    visible: bool = True
    color: MoleculeColormap | str = "lime"
    point_size: float = 4.2

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @classmethod
    def from_layer(
        cls,
        gui: "CylindraMainWidget",
        layer: "MoleculesLayer",
        ext: str,
    ) -> "MoleculesInfo":
        try:
            _src = gui.tomogram.splines.index(layer.source_component)
        except ValueError:
            _src = None
        info = layer.colormap_info
        if isinstance(info, str):
            color = info
        else:
            color = MoleculeColormap(
                cmap=info.to_list(),
                limits=info.clim,
                by=info.name,
            )
        return cls(
            name=f"{layer.name}{ext}",
            source=_src,
            visible=layer.visible,
            color=color,
            point_size=layer.point_size,
        )

    def to_molecules(self, project_dir: Path) -> "Molecules":
        from acryo import Molecules

        path = project_dir / self.name
        if not path.exists():
            warnings.warn(
                f"Cannot find molecule file {path}. Probably it was moved?",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        return Molecules.from_file(path)

    def to_layer(self, gui: "CylindraMainWidget", project_dir: Path):
        from cylindra._napari import MoleculesLayer

        mole = self.to_molecules(project_dir)

        if self.source is not None:
            src = gui.tomogram.splines[self.source]
        else:
            src = None
        kwargs = {
            "name": self.stem,
            "source": src,
            "visible": self.visible,
            "size": self.point_size,
        }
        match self.color:
            case str(color):
                kwargs["face_color"] = color
            case cmap:
                kwargs["cmap"] = cmap.model_dump()

        return MoleculesLayer.construct(mole, **kwargs)

    def save_layer(self, gui: "CylindraMainWidget", dir: Path):
        layer = gui.mole_layers[self.stem]
        return layer.molecules.to_file(dir / f"{self.name}")


class LandscapeInfo(LayerInfo):
    name: str = "#unknown"  # including extension
    source: int | None = None
    visible: bool = True
    level: float | None

    @classmethod
    def from_layer(
        cls, gui: "CylindraMainWidget", layer: "LandscapeSurface"
    ) -> "LayerInfo":
        try:
            _src = gui.tomogram.splines.index(layer.source_component)
        except ValueError:
            _src = None
        return LandscapeInfo(
            name=layer.name,
            source=_src,
            visible=layer.visible,
            level=layer.level,
        )

    def to_layer(self, gui: "CylindraMainWidget", project_dir: Path):
        from cylindra._napari import LandscapeSurface
        from cylindra.components.landscape import Landscape

        path = project_dir / self.name
        if not path.exists():
            warnings.warn(
                f"Cannot find landscape file {path}. Probably it was moved?",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        landscape = Landscape.from_dir(path)

        if self.source is not None:
            src = gui.tomogram.splines[self.source]
        else:
            src = None
        layer = LandscapeSurface(
            landscape, level=self.level, name=self.name, visible=self.visible
        )
        layer.source_component = src
        return layer

    def save_layer(self, gui: "CylindraMainWidget", dir: Path):
        layer: LandscapeSurface = gui.parent_viewer.layers[self.name]
        return layer.landscape.save(dir / f"{self.name}")
