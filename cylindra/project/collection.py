from typing import Any, Union, TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel
from cylindra.const import PropertyNames as H, get_versions
from .sequence import ProjectSequence
from ._base import BaseProject, PathLike

if TYPE_CHECKING:
    from cylindra.widgets.collection import ProjectCollectionWidget

    
class CylindraCollectionProject(BaseProject):
    datetime: str
    version: str
    dependency_versions: dict[str, str]
    macro: PathLike
    children: list[PathLike]  # list of project paths
    filter_predicate: Union[str, None] = None
    project_path: Union[Path, None] = None

    @classmethod
    def from_gui(cls, gui: "ProjectCollectionWidget", json_path: Path) -> "CylindraCollectionProject":
        """Create a project collection from a widget."""
        from datetime import datetime
        
        _versions = get_versions()
        
        # Save path of macro
        root = json_path.parent
        macro_path = root / f"{json_path.stem}-script.py"
        
        def as_relative(p: Path):
            try:
                out = p.relative_to(root)
            except Exception:
                out = p
            return out

        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version = next(iter(_versions.values())),
            dependency_versions = _versions,
            macro=as_relative(macro_path),
            children=[as_relative(fp) for fp in gui._get_project_paths()],
            filter_predicate=str(gui.collection.FilterExpr.filter_expression),
        )
    
    def to_sequence(self) -> "ProjectSequence":
        seq = ProjectSequence()
        for path in self.children:
            seq.add(path)
        return seq
        
    def to_gui(self, gui: "ProjectCollectionWidget") -> None:
        """Load a project collection into a widget."""
        gui.set_sequence(self.to_sequence())
        return None

    @classmethod
    def save_gui(cls, gui: "ProjectCollectionWidget", json_path: PathLike) -> None:
        """Save a project collection from a widget."""
        json_path = Path(json_path)
        self = cls.from_gui(gui, json_path)
        self.to_json(json_path)
        macro_path = json_path.parent / str(self.macro)
        macro_path.write_text(str(gui.macro))
        return None

    def resolve_path(self, file_dir: PathLike) -> None:
        """Resolve paths."""
        self.macro = Path(self.macro).resolve(file_dir)
        self.children = [Path(p).resolve(file_dir) for p in self.children]
        return None
