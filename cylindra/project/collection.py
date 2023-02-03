import os
import json
from typing import Any, Union, TYPE_CHECKING
from pathlib import Path
from cylindra.const import PropertyNames as H
from ._base import BaseProject, PathLike

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget

class ProjectCollection(BaseProject):
    
    datetime: str
    version: str
    dependency_versions: dict[str, str]
    macro: PathLike
    
    project_path: Union[Path, None] = None
