from __future__ import annotations
from typing import Iterable
import numpy as np
import napari
from ..components import Molecules, MtSpline
from ..const import MOLECULES, SOURCE

def add_molecules(viewer: napari.Viewer, mol: Molecules, name, source: MtSpline = None):
    """Add Molecules object as a point layer."""
    metadata ={MOLECULES: mol}
    if source is not None:
        metadata.update({SOURCE: source})
    points_layer = viewer.add_points(
        mol.pos, size=3, face_color="lime", edge_color="lime",
        out_of_slice_display=True, name=name, metadata=metadata
        )
    
    points_layer.shading = "spherical"
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
    viewer.dims.current_step = list(center.astype(int))
    return None
