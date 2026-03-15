from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import starfile
except ImportError:

    class _starfile_module:
        def __getattr__(self, name):
            raise ImportError(
                "The 'starfile' package is required for RELION I/O functions. "
                "Please install it via 'pip install starfile'."
            )

    starfile = _starfile_module()


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
        return starfile.read(job_star_path, always_dict=True)["job"]["rlnJobTypeLabel"]
    raise ValueError(f"{job_dir} is not a RELION job folder.")


def shape_to_center_zyx(shape: tuple[int, int, int], scale: float) -> np.ndarray:
    return (np.array(shape) / 2 - 1) * scale


def strip_relion5_prefix(name: str):
    """Strip the RELION 5.0 "rec" prefix from the name."""
    if name.startswith("rec_"):
        name = name[4:]
    if "." in name:
        name = name.split(".")[0]
    return name
