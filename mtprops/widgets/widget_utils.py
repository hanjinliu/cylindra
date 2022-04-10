from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import numpy as np
from scipy import ndimage as ndi
import napari
from napari.layers import Points, Vectors, Tracks, Labels
from ..components import Molecules, MtSpline
from ..const import MOLECULES, GVar, Mole

class FileFilter(SimpleNamespace):
    """File dialog filter strings"""
    
    IMAGE = "Tomograms (*.mrc;*.rec;*.tif;*.tiff;*.map);;All files (*)"
    JSON = "JSON(*.json;*.txt);;All files (*)"
    CSV = "CSV(*.csv;*.txt);*.dat;;All files (*)"


def add_molecules(viewer: napari.Viewer, mol: Molecules, name):
    """Add Molecules object as a point layer."""
    metadata ={MOLECULES: mol}
    points_layer = viewer.add_points(
        mol.pos, 
        size=3,
        face_color="lime",
        edge_color="lime",
        out_of_slice_display=True,
        name=name,
        metadata=metadata,
        features=mol.features,
    )
    
    points_layer.shading = "spherical"
    points_layer.editable = False
    return points_layer


def change_viewer_focus(
    viewer: "napari.Viewer",
    center: Iterable[float],
    scale: float = 1.0,
) -> None:
    center = np.asarray(center)
    viewer.camera.center = center * scale
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.current_step = list(np.round(center*scale).astype(int))
    return None

def update_features(
    layer: Points | Vectors | Tracks | Labels,
    values: dict[str, Iterable] = None,
    **kwargs,
):
    """Update layer features with new values."""
    features = layer.features
    if values is not None:
        if kwargs:
            raise ValueError("Cannot specify both values and kwargs.")
        kwargs = values
    for name, value in kwargs.items():
        features[name] = value
    layer.features = features
    if MOLECULES in layer.metadata:
        mole: Molecules = layer.metadata[MOLECULES]
        mole.features = features
    return None

def molecules_to_spline(layer: Points) -> MtSpline:
    """Convert well aligned molecule positions into a spline."""
    mole: Molecules = layer.metadata[MOLECULES]
    spl = MtSpline(degree=GVar.splOrder)
    npf = int(round(np.max(mole.features[Mole.pf]) + 1))
    all_coords = mole.pos.reshape(-1, npf, 3)
    mean_coords = np.mean(all_coords, axis=1)
    spl.fit(mean_coords, variance=GVar.splError**2)
    return spl

def y_coords_to_start_number(u: np.ndarray, npf: int):
    """infer start number using the y coordinates in spline coordinate system."""
    a0 = (u[-npf] - u[0]) / (u.size - npf)
    a1 = np.mean((u[::npf] - u[(npf-1)::npf])/(npf - 1))
    return int(round(a1/a0))

def coords_to_params(pos: np.ndarray, npf: int) -> tuple[float, float, float]:
    """infer pitch length using the y coordinates in spline coordinate system."""
    ndim = 3
    pos = pos.reshape(-1, npf, ndim)
    ypitch = np.mean(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=2)))
    lateral_spacing = np.mean(np.sqrt(np.sum(np.diff(pos, axis=1)**2, axis=2)))
    radius = lateral_spacing * npf / 2 / np.pi
    return ypitch, lateral_spacing, radius

def get_versions() -> dict[str, str]:
    """Return version info of relevant libraries."""
    import napari
    import impy as ip
    import magicgui
    from .. import __version__
    import magicclass as mcls
    import dask
    
    return {
        "MTProps": __version__,
        "numpy": np.__version__,
        "impy": ip.__version__,
        "magicgui": magicgui.__version__,
        "magicclass": mcls.__version__,
        "napari": napari.__version__,
        "dask": dask.__version__,
    }

def sheared_heatmap(arr: np.ndarray, npf: int = 13, start: int = 3):
    sy, sx = arr.shape
    ny, nx = 5, 10
    arr1 = np.stack([arr]*ny, axis=1).reshape(sy * ny, sx)
    arr2 = np.stack([arr1]*nx, axis=2).reshape(sy * ny, sx * nx)
    shear = start / npf * ny / nx
    mtx = np.array(
        [[1., shear, 0.],
         [0., 1., 0.],
         [0., 0., 1.]]
    )
    return ndi.affine_transform(arr2, matrix=mtx, order=1, prefilter=False)

def resolve_path(path: str | Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    path_joined = root / path
    if path_joined.exists():
        return path_joined
    raise ValueError(f"Path {path} could not be resolved.")