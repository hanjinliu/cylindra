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

from magicgui.widgets import SpinBox, Container
from magicgui.types import Undefined
from magicclass.widgets import ScrollableContainer
from magicclass.logging import getLogger
import napari

from acryo import Molecules, TomogramSimulator
from cylindra import utils
from cylindra.const import MoleculesHeader as Mole, nm, GlobalVariables as GVar
from cylindra.types import MoleculesLayer
from cylindra.components._base import BaseComponent

if TYPE_CHECKING:
    from cylindra.components import CylTomogram


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
        size=GVar.pointSize,
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


def plot_forward_and_reverse(
    template_fw, fit_fw, zncc_fw, template_rv, fit_rv, zncc_rv
):
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


@dataclass
class FscResult:
    """Result of Fourier Shell Correlation."""

    freq: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    resolution_0143: nm
    resolution_0500: nm


def extend_protofilament(
    mole: Molecules, counts: dict[int, tuple[int, int]]
) -> Molecules:
    """Extend a protofilament by linearly outpolation."""
    existing_pf_id = set(mole.features[Mole.pf].unique())
    if not counts.keys() <= existing_pf_id:
        raise ValueError(f"Invalid ID: {counts.keys() - existing_pf_id}")
    df = mole.to_dataframe()
    zyxp = [Mole.z, Mole.y, Mole.x, Mole.position]
    predictables = [
        Mole.z,
        Mole.y,
        Mole.x,
        Mole.zvec,
        Mole.yvec,
        Mole.xvec,
        Mole.position,
        Mole.nth,
        Mole.pf,
    ]
    schema = {c: df[c].dtype for c in predictables}
    to_prepend: list[Molecules] = []
    to_append: list[Molecules] = []
    for _pf_id, (_n_prepend, _n_append) in counts.items():
        # get the data frame of the protofilament
        df_filt = df.filter(pl.col(Mole.pf) == _pf_id).sort(pl.col(Mole.position))
        prepend_start = df_filt[0]
        prepend_vec = prepend_start.select(zyxp) - df_filt[1].select(zyxp)
        append_start = df_filt[-1]
        append_vec = append_start.select(zyxp) - df_filt[-2].select(zyxp)
        nth_start = prepend_start[Mole.nth][0]
        nth_stop = append_start[Mole.nth][0]

        rng = np.arange(_n_prepend, 0, -1)
        df_prepend = pl.DataFrame(
            {
                Mole.z: prepend_start[Mole.z][0] + prepend_vec[Mole.z][0] * rng,
                Mole.y: prepend_start[Mole.y][0] + prepend_vec[Mole.y][0] * rng,
                Mole.x: prepend_start[Mole.x][0] + prepend_vec[Mole.x][0] * rng,
                Mole.zvec: np.full(_n_prepend, prepend_start[Mole.zvec][0]),
                Mole.yvec: np.full(_n_prepend, prepend_start[Mole.yvec][0]),
                Mole.xvec: np.full(_n_prepend, prepend_start[Mole.xvec][0]),
                Mole.position: prepend_start[Mole.position][0]
                + prepend_vec[Mole.position][0] * rng,  # fmt: skip
                Mole.nth: np.arange(nth_start - _n_prepend, nth_start),
                Mole.pf: np.full(_n_prepend, _pf_id),
            },
            schema=schema,
        )

        rng = np.arange(1, _n_append + 1)
        df_append = pl.DataFrame(
            {
                Mole.z: append_start[Mole.z][0] + append_vec[Mole.z][0] * rng,
                Mole.y: append_start[Mole.y][0] + append_vec[Mole.y][0] * rng,
                Mole.x: append_start[Mole.x][0] + append_vec[Mole.x][0] * rng,
                Mole.zvec: np.full(_n_append, append_start[Mole.zvec][0]),
                Mole.yvec: np.full(_n_append, append_start[Mole.yvec][0]),
                Mole.xvec: np.full(_n_append, append_start[Mole.xvec][0]),
                Mole.position: append_start[Mole.position][0]
                + append_vec[Mole.position][0] * rng,  # fmt: skip
                Mole.nth: np.arange(nth_stop + 1, nth_stop + _n_append + 1),
                Mole.pf: np.full(_n_append, _pf_id),
            },
            schema=schema,
        )
        if df_prepend.shape[0] > 0:
            to_prepend.append(Molecules.from_dataframe(df_prepend))
        if df_append.shape[0] > 0:
            to_append.append(Molecules.from_dataframe(df_append))
    return Molecules.concat(to_prepend + [mole] + to_append)


class ProtofilamentEdit(ScrollableContainer[Container[SpinBox]]):
    def __init__(
        self, value=Undefined, *, labels=True, nullable=False, **kwargs
    ) -> None:
        super().__init__(labels=labels, **kwargs)
        self.value = value

    def _add_row(self, label: int, value: tuple[int, int]):
        val0, val1 = value
        row = Container(
            widgets=[
                SpinBox(value=val0, tooltip="Number of molecules to prepend"),
                SpinBox(value=val1, tooltip="Number of molecules to append"),
            ],
            layout="horizontal",
            label=str(label),
        )
        row.margins = (0, 0, 0, 0)
        self.append(row)
        row.changed.disconnect()
        row.changed.connect(self._on_changed)

    @property
    def value(self) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for row in self:
            pf_id = int(row.label)
            vals = row[0].value, row[1].value
            out[pf_id] = vals
        return out

    @value.setter
    def value(self, val: dict[int, tuple[int, int]]):
        if val is Undefined:
            val = {}
        with self.changed.blocked():
            self.clear()
            for k, v in val.items():
                self._add_row(k, v)
        self.changed.emit(val)

    def _on_changed(self):
        self.changed.emit(self.value)


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
