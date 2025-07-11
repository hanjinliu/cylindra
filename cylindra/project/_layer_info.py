import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from acryo import Molecules
    from napari.layers import Layer

    from cylindra._napari import InteractionVector, LandscapeSurface, MoleculesLayer
    from cylindra.widgets.main import CylindraMainWidget


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
            _warn_not_exist(path)
            return None
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
            _warn_not_exist(path)
            return None
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


class InteractionInfo(LayerInfo):
    name: str = "#unknown"  # including extension
    visible: bool = True
    width: float = 0.7
    origin: str | None = None
    target: str | None = None

    @classmethod
    def from_layer(
        cls, gui: "CylindraMainWidget", layer: "InteractionVector"  # noqa: ARG003
    ) -> "LayerInfo":
        origin = target = None
        for mole_layer in gui.mole_layers:
            if mole_layer.molecules is layer.net.molecules_origin:
                origin = mole_layer.name
            if mole_layer.molecules is layer.net.molecules_target:
                target = mole_layer.name
        return InteractionInfo(
            name=layer.name,
            visible=layer.visible,
            width=layer.edge_width,
            origin=origin,
            target=target,
        )

    def to_layer(self, gui: "CylindraMainWidget", project_dir: Path):  # noqa: ARG003
        from cylindra._napari import InteractionVector
        from cylindra.components.interaction import InterMoleculeNet

        path = project_dir / self.name
        if not path.exists():
            _warn_not_exist(path)
            return None
        net = InterMoleculeNet.from_dir(path)
        # if reference found, replace it with the one in the gui
        existing_names = gui.mole_layers.names()
        if self.origin is not None and self.origin in existing_names:
            net.molecules_origin = gui.mole_layers[self.origin].molecules
        if self.target is not None and self.target in existing_names:
            net.molecules_target = gui.mole_layers[self.target].molecules
        layer = InteractionVector(
            net, edge_width=self.width, name=self.name, visible=self.visible
        )
        return layer

    def save_layer(self, gui: "CylindraMainWidget", dir: Path):
        layer: InteractionVector = gui.parent_viewer.layers[self.name]
        return layer.net.save(dir / f"{self.name}")


def _warn_not_exist(path):
    return warnings.warn(
        f"Cannot find landscape file {path}. Probably it was moved?",
        RuntimeWarning,
        stacklevel=2,
    )
