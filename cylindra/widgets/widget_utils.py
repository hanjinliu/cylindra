from __future__ import annotations
from types import SimpleNamespace
from typing import Sequence, TYPE_CHECKING
from dataclasses import dataclass
from timeit import default_timer
import inspect

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
import polars as pl

from magicclass.logging import getLogger
import napari

from acryo import Molecules, TomogramSimulator
from cylindra import utils
from cylindra.const import MoleculesHeader as Mole, nm, GlobalVariables as GVar
from cylindra.types import MoleculesLayer
from cylindra.components._base import BaseComponent

if TYPE_CHECKING:
    from cylindra.components import CylTomogram, CylSpline


# namespace used in predicate
POLARS_NAMESPACE = {
    "pl": pl,
    "int": int,
    "float": float,
    "str": str,
    "np": np,
    "__builtins__": {},
}


class FileFilter(SimpleNamespace):
    """File dialog filter strings"""

    IMAGE = "Tomograms (*.mrc;*.rec;*.tif;*.tiff;*.map);;All files (*)"
    JSON = "JSON(*.json;*.txt);;All files (*)"
    PROJECT = "Project file (project.json);;JSON (*.json;*.txt);;All files (*)"
    CSV = "CSV (*.csv;*.txt;*.dat);;All files (*)"
    PY = "Python (*.py);;All files (*)"
    MOD = "Model files (*.mod);;All files (*.txt;*.csv)"


class timer:
    def __init__(self, name: str | None = None):
        if name is None:
            name = inspect.stack()[0].function
        self.name = name
        self.start = default_timer()

    def toc(self):
        getLogger("cylindra").print_html(
            f"<code>{self.name}</code> ({default_timer() - self.start:.1f} sec)"
        )


def add_molecules(
    viewer: napari.Viewer,
    mol: Molecules,
    name: str,
    source: BaseComponent | None = None,
) -> MoleculesLayer:
    """Add Molecules object as a point layer."""
    layer = MoleculesLayer(
        mol,
        size=GVar.point_size,
        face_color="lime",
        edge_color="lime",
        out_of_slice_display=True,
        name=name,
    )
    if source is not None:
        layer.source_component = source
    viewer.add_layer(layer)
    layer.shading = "spherical"
    layer.editable = False
    return layer


def change_viewer_focus(
    viewer: napari.Viewer,
    center: Sequence[float],
    scale: float = 1.0,
) -> None:
    center = np.asarray(center)
    v_scale = np.array([r[2] for r in viewer.dims.range])
    viewer.camera.center = center * scale
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.set_current_step(axis=0, value=center[0] / v_scale[0] * scale)
    return None


def sheared_heatmap(arr: np.ndarray, npf: int = 13, start: int = 3):
    sy, sx = arr.shape
    ny, nx = 5, 10
    arr1 = np.stack([arr] * ny, axis=1).reshape(sy * ny, sx)
    arr2 = np.stack([arr1] * nx, axis=2).reshape(sy * ny, sx * nx)
    shear = start / npf * ny / nx
    mtx = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return ndi.affine_transform(arr2, matrix=mtx, order=1, prefilter=False)


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
    plt.xticks(np.arange(0, 2 * npf + 1, 4))
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

    ind = freq <= 0.7
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
        alpha=0.3,
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
    freq: np.ndarray, fsc: np.ndarray, crit: float = 0.143, scale: nm = 1.0
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
            f0 = freq[i - 1]
            f1 = freq[i]
            fsc0 = fsc[i - 1]
            freq0 = (crit - fsc1) / (fsc0 - fsc1) * (f0 - f1) + f1
            resolution = scale / freq0
            break
    else:
        resolution = np.nan
    return resolution


def plot_projections(merge: np.ndarray):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5))
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


def plot_forward_and_reverse(
    template_fw, fit_fw, zncc_fw, template_rv, fit_rv, zncc_rv
):
    import matplotlib.pyplot as plt
    from cylindra import utils

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


@dataclass
class FscResult:
    """Result of Fourier Shell Correlation."""

    freq: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    resolution_0143: nm
    resolution_0500: nm


def coordinates_with_extensions(
    spl: CylSpline, n_extend: dict[int, tuple[int, int]]
) -> NDArray[np.int32]:
    model = spl.cylinder_model()
    coords = list[NDArray[np.int32]]()
    ny, npf = model.shape
    for _idx in range(npf):
        _append, _prepend = n_extend.get(_idx, (0, 0))
        if ny + _append + _prepend <= 0:
            continue  # size is zero
        nth = np.arange(-_prepend, ny + _append, dtype=np.int32)
        npf = np.full(nth.size, _idx, dtype=np.int32)
        coords.append(np.stack([nth, npf], axis=1))

    return np.concatenate(coords, axis=0)


class PaintDevice:
    """
    Device used for painting 3D images.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the target image.
    scale : nm
        Scale of the target image.
    """

    def __init__(self, shape: tuple[int, int, int], scale: nm):
        self._shape = shape
        self._scale = scale

    @property
    def scale(self) -> nm:
        return self._scale

    def paint_molecules(self, template: NDArray[np.float32], molecules: Molecules):
        simulator = TomogramSimulator(order=1, scale=self._scale)
        simulator.add_molecules(molecules, template)
        img = simulator.simulate(self._shape)
        return img

    def paint_cylinders(self, ft_size: nm, tomo: CylTomogram):
        lbl = np.zeros(self._shape, dtype=np.uint8)
        all_df = tomo.collect_localprops()
        if all_df is None:
            raise ValueError("No local property found.")
        bin_scale = self._scale  # scale of binned reference image
        binsize = utils.roundint(bin_scale / tomo.scale)

        lz, ly, lx = (
            utils.roundint(r / bin_scale * 1.73) * 2 + 1 for r in [15, ft_size / 2, 15]
        )
        center = np.array([lz, ly, lx]) / 2 + 0.5
        z, y, x = np.indices((lz, ly, lx))
        cylinders = []
        matrices = []
        for i, spl in enumerate(tomo.splines):
            # Prepare template hollow image
            r0 = spl.radius / tomo.scale * 0.9 / binsize
            r1 = spl.radius / tomo.scale * 1.1 / binsize
            _sq = (z - lz / 2 - 0.5) ** 2 + (x - lx / 2 - 0.5) ** 2
            domains = []
            dist = [-np.inf] + list(spl.distances()) + [np.inf]
            for j in range(spl.anchors.size):
                domain = (r0**2 < _sq) & (_sq < r1**2)
                ry = (
                    min(
                        abs(dist[j + 1] - dist[j]) / 2,
                        abs(dist[j + 2] - dist[j + 1]) / 2,
                        ft_size / 2,
                    )
                    / bin_scale
                    + 0.5
                )

                ry = max(utils.ceilint(ry), 1)
                domain[:, : ly // 2 - ry] = 0
                domain[:, ly // 2 + ry + 1 :] = 0
                domain = domain.astype(np.float32)
                domains.append(domain)

            cylinders.append(domains)
            matrices.append(spl.affine_matrix(center=center, inverse=True))
            yield

        cylinders = np.concatenate(cylinders, axis=0)
        matrices = np.concatenate(matrices, axis=0)

        out = np.empty_like(cylinders)
        for i, (img, matrix) in enumerate(zip(cylinders, matrices)):
            out[i] = ndi.affine_transform(img, matrix, order=1, cval=0, prefilter=False)
        out = out > 0.3

        # paint roughly
        for i, crd in enumerate(tomo.collect_anchor_coords()):
            center = tomo.nm2pixel(crd, binsize=binsize)
            sl: list[slice] = []
            outsl: list[slice] = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = utils.make_slice_and_pad(c - l // 2, c + l // 2 + 1, size)
                sl.append(_sl)
                outsl.append(
                    slice(
                        _pad[0] if _pad[0] > 0 else None,
                        -_pad[1] if _pad[1] > 0 else None,
                    )
                )

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl[sl][out[i][outsl]] = i + 1
            yield

        return lbl
