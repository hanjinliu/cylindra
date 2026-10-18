from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

try:
    import starfile_rs
except ImportError:

    class _starfile_rs_module:
        def __getattr__(self, name):
            raise ImportError(
                "The 'starfile-rs' package is required for RELION I/O functions. "
                "Please install it via 'pip install starfile-rs'."
            )

    starfile_rs = _starfile_rs_module()


def get_optimisation_set_star(job_dir_path: Path) -> Path:
    if fp := _relion_job_get_last(job_dir_path, "run_it*_optimisation_set.star"):
        return fp
    raise ValueError(
        f"No optimisation set star files found in {job_dir_path}. "
        "Please ensure at least one iteration has finished."
    )


def get_run_data_star(job_dir_path: Path) -> Path | None:
    return _relion_job_get_last(job_dir_path, "run_it*_data.star")


def _relion_job_get_last(job_dir_path: Path, pattern: str) -> Path | None:
    path_list = sorted(job_dir_path.glob(pattern), key=lambda p: p.stem)
    if len(path_list) == 0:
        return None
    return path_list[-1]


def relion_project_path(path: Path) -> Path:
    return path.parent.parent


def get_job_type(job_dir: Path) -> str:
    """Determine the type of RELION job based on the directory structure."""
    if (job_star_path := job_dir / "job.star").exists():
        star = starfile_rs.read_star(job_star_path)
        return star["job"].trust_single()["rlnJobTypeLabel"]
    raise ValueError(f"{job_dir} is not a RELION job folder.")


def read_loop_block(path: Path, block_name: str = "particles") -> pl.DataFrame:
    """Read a loop block as a DataFrame (the only block, or `block_name` if many)."""
    star = starfile_rs.read_star(path)
    if len(star) == 1:
        return star.first().to_polars()
    return star[block_name].to_polars()


def shape_to_center_zyx(shape: tuple[int, int, int], scale: float) -> np.ndarray:
    return (np.array(shape) / 2 - 1) * scale


def strip_relion5_prefix(name: str):
    """Strip the RELION 5.0 "rec" prefix from the name."""
    if name.startswith("rec_"):
        name = name[4:]
    if "." in name:
        name = name.split(".")[0]
    return name
