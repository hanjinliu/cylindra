from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, TYPE_CHECKING
import numpy as np
from scipy import ndimage as ndi
import napari
from napari.layers import Points, Vectors, Tracks, Labels
from cylindra.components import CylSpline
from cylindra.const import MOLECULES, GlobalVariables as GVar, MoleculesHeader as Mole

if TYPE_CHECKING:
    from acryo import Molecules
    from cylindra.const import nm

class FileFilter(SimpleNamespace):
    """File dialog filter strings"""
    
    IMAGE = "Tomograms (*.mrc;*.rec;*.tif;*.tiff;*.map);;All files (*)"
    JSON = "JSON(*.json;*.txt);;All files (*)"
    CSV = "CSV(*.csv;*.txt);*.dat;;All files (*)"
    PY = "Python (*.py);;All files (*)"


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
    center: Sequence[float],
    scale: float = 1.0,
) -> None:
    center = np.asarray(center)
    v_scale = np.array([r[2] for r in viewer.dims.range])
    viewer.camera.center = center * scale
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.set_current_step(axis=0, value=center[0]/v_scale[0]*scale)
    return None

def update_features(
    layer: Points | Vectors | Tracks | Labels,
    values: dict[str, Sequence] = None,
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

def molecules_to_spline(layer: Points) -> CylSpline:
    """Convert well aligned molecule positions into a spline."""
    mole: Molecules = layer.metadata[MOLECULES]
    spl = CylSpline(degree=GVar.splOrder)
    npf = int(round(np.max(mole.features[Mole.pf]) + 1))
    all_coords = mole.pos.reshape(-1, npf, 3)
    mean_coords = np.mean(all_coords, axis=1)
    spl.fit_coa(mean_coords, min_radius=GVar.minCurvatureRadius)
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
    from .._info import __version__
    import magicclass as mcls
    import dask
    
    return {
        "cylindra": __version__,
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


def layer_to_coordinates(layer: Points, npf: int | None = None):
    """Convert point coordinates of a Points layer into a structured array."""
    if npf is None:
        npf = layer.features[Mole.pf].max() + 1
    data = layer.data.reshape(-1, npf, 3)
    import impy as ip

    data = ip.asarray(data, name=layer.name, axes=["L", "PF", "dim"])
    data.axes["dim"].labels = ("z", "y", "x")
    return data


def plot_seam_search_result(score: np.ndarray, npf: int):
    import matplotlib.pyplot as plt
    imax = np.argmax(score)
    # plot the score
    plt.figure(figsize=(6, 2.4))
    plt.axvline(imax, color="gray", alpha=0.6)
    plt.axhline(score[imax], color="gray", alpha=0.6)
    plt.plot(score)
    plt.xlabel("PF position")
    plt.ylabel("ΔCorr")
    plt.xticks(np.arange(0, 2*npf+1, 4))
    plt.title("Score")
    plt.tight_layout()
    plt.show()

def plot_fsc(
    freq: np.ndarray,
    fsc_mean: np.ndarray,
    fsc_std: np.ndarray,
    crit: list[float],
    scale: nm,
):
    import matplotlib.pyplot as plt
    ind = (freq <= 0.7)
    plt.axhline(0.0, color="gray", alpha=0.5, ls="--")
    plt.axhline(1.0, color="gray", alpha=0.5, ls="--")
    for cr in crit:
        plt.axhline(cr, color="violet", alpha=0.5, ls="--")
    plt.plot(freq[ind], fsc_mean[ind], color="gold")
    plt.fill_between(
        freq[ind],
        y1=fsc_mean[ind] - fsc_std[ind],
        y2=fsc_mean[ind] + fsc_std[ind],
        color="gold",
        alpha=0.3
    )
    plt.xlabel("Spatial frequence (1/nm)")
    plt.ylabel("FSC")
    plt.ylim(-0.1, 1.1)
    xticks = np.linspace(0, 0.7, 8)
    per_nm = [r"$\infty$"] + [f"{x:.2f}" for x in scale / xticks[1:]]
    plt.xticks(xticks, per_nm)
    plt.tight_layout()
    plt.show()

def calc_resolution(
    freq: np.ndarray,
    fsc: np.ndarray,
    crit: float = 0.143,
    scale: nm = 1.0
) -> nm:
    """
    Calculate resolution using arrays of frequency and FSC.
    This function uses linear interpolation to find the solution.
    If the inputs are not accepted, 0 will be returned.
    """
    freq0 = None
    for i, fsc1 in enumerate(fsc):
        if fsc1 < crit:
            if i == 0:
                resolution = 0
                break
            f0 = freq[i-1]
            f1 = freq[i]
            fsc0 = fsc[i-1]
            freq0 = (crit - fsc1)/(fsc0 - fsc1) * (f0 - f1) + f1
            resolution = scale / freq0
            break
    else:
        resolution = 0
    return resolution

def plot_projections(merge: np.ndarray):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2)
    # normalize
    if merge.dtype.kind == "f":
        merge = np.clip(merge, 0, 1)
    elif merge.dtype.kind in "ui":
        merge = np.clip(merge, 0, 255)
    else:
        raise RuntimeError("dtype not supported.")
    axes[0].imshow(np.max(merge, axis=0))
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[1].imshow(np.max(merge, axis=1))
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    plt.tight_layout()
    plt.show()
    return None

def plot_forward_and_reverse(template_fw, fit_fw, zncc_fw, template_rv, fit_rv, zncc_rv):
    import matplotlib.pyplot as plt
    from .. import utils
    fig, axes = plt.subplots(nrows=2, ncols=3)
    template_proj_fw = np.max(template_fw, axis=1)
    fit_proj_fw = np.max(fit_fw, axis=1)
    merge_fw = utils.merge_images(fit_proj_fw, template_proj_fw)
    template_proj_rv = np.max(template_rv, axis=1)
    fit_proj_rv = np.max(fit_rv, axis=1)
    merge_rv = utils.merge_images(fit_proj_rv, template_proj_rv)
    axes[0][0].imshow(template_proj_fw, cmap="gray")
    axes[0][1].imshow(fit_proj_fw, cmap="gray")
    axes[0][2].imshow(merge_fw)
    axes[0][0].text(0, 0, f"{zncc_fw:.3f}", va="top", fontsize=14)
    axes[1][0].imshow(template_proj_rv, cmap="gray")
    axes[1][1].imshow(fit_proj_rv, cmap="gray")
    axes[1][2].imshow(merge_rv)
    axes[1][0].text(0, 0, f"{zncc_rv:.3f}", va="top", fontsize=14)
    axes[0][0].set_title("Template")
    axes[0][1].set_title("Average")
    axes[0][2].set_title("Merge")
    axes[0][0].set_ylabel("Forward")
    axes[1][0].set_ylabel("Reverse")
    
    return None
