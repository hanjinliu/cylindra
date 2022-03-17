from __future__ import annotations
from types import SimpleNamespace
from typing import Iterable
import numpy as np
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
        mol.pos, size=3, face_color="lime", edge_color="lime",
        out_of_slice_display=True, name=name, metadata=metadata
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
    feature_name: str,
    value: Iterable,
):
    """Update layer features with new values."""
    features = layer.features
    features[feature_name] = value
    layer.features = features
    return None

def molecules_to_spline(layer: Points) -> MtSpline:
    """Convert well aligned molecule positions into a spline."""
    mole: Molecules = layer.metadata[MOLECULES]
    spl = MtSpline(degree=GVar.splOrder)
    npf = int(round(np.max(layer.features[Mole.pf]) + 1))
    all_coords = mole.pos.reshape(-1, npf, 3)
    mean_coords = np.mean(all_coords, axis=1)
    spl.fit(mean_coords, variance=GVar.splError**2)
    return spl

def y_coords_to_start_number(u: np.ndarray, npf: int):
    """infer start number using the y coordinates in spline coordinate system."""
    a0 = (u[-npf] - u[0]) / (u.size - npf)
    a1 = np.mean((u[::npf] - u[(npf-1)::npf])/(npf - 1))
    return int(round(a1/a0))

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