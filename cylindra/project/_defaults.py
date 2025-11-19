import glob
from pathlib import Path
from typing import Self

import tomllib
from pydantic import BaseModel, field_validator

from cylindra.const import ImageFilter
from cylindra.project._base import MissingWedge

TOMOGRAM_DEFAULTS_PATH_NAME = ".cylindra-defaults.toml"

# If file ".cylindra-defaults.toml" exists in the same directory as the tomogram, this
# file is used to load default settings for reading this tomogram.


class TomogramDefaults(BaseModel):
    scale: float | None = None
    invert: bool = False
    image_reference: str | None = None
    """Absolute or relative path to the reference image file. Can use format string
    to include tomogram stem."""
    invert_reference: bool = False
    missing_wedge: MissingWedge | None = None
    bin_size: list[int] | None = None
    filter: ImageFilter | None = None

    @classmethod
    def from_dir(cls, dirpath: str | Path, parent_ok: bool = True) -> Self | None:
        dirpath = Path(dirpath)
        toml_path = dirpath / TOMOGRAM_DEFAULTS_PATH_NAME
        if parent_ok:  # search in parent directories
            while not toml_path.exists() and dirpath.parent != dirpath:
                dirpath = dirpath.parent
                toml_path = dirpath / TOMOGRAM_DEFAULTS_PATH_NAME
        if not toml_path.exists():
            return None
        toml = tomllib.loads(toml_path.read_text())
        if mw_dict := toml.pop("missing_wedge"):
            mw = MissingWedge.parse(mw_dict)
            toml["missing_wedge"] = mw
        return cls.model_validate(toml)

    @field_validator("scale")
    def _check_scale(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Scale must be positive.")
        return v

    @field_validator("bin_size")
    def _check_bin_size(cls, v):
        if v is not None:
            for b in v:
                if b <= 0 or not isinstance(b, int):
                    raise ValueError("Bin sizes must be positive integers.")
        return v

    def resolve_reference_path(self, tomogram_path: str | Path) -> Path | None:
        if self.image_reference is None:
            return None
        tomogram_path = Path(tomogram_path)
        if "{" in self.image_reference:
            ref_path = Path(self.image_reference.format(tomogram_path.stem))
        else:
            ref_path = Path(self.image_reference)
        if ref_path.is_absolute():
            return _resolve_autofill(ref_path)
        ref_path_parts = list(ref_path.parts)
        root = tomogram_path.parent
        while ref_path_parts and ref_path_parts[0] in (".", ".."):
            part = ref_path_parts.pop(0)
            if part == ".":
                continue
            elif part == "..":
                root = root.parent
        return _resolve_autofill(root / Path(*ref_path_parts))


def _resolve_autofill(path: Path) -> Path | None:
    pattern = str(path)
    if "*" in pattern or "?" in pattern:
        if matched_path := next(reversed(glob.glob(pattern)), None):
            return Path(matched_path)
    return None
