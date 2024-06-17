from __future__ import annotations

from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    overload,
)

import impy as ip
import numpy as np
from acryo import SubtomogramLoader
from acryo.tilt import TiltSeriesModel
from dask import array as da
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.fft import fft2, ifft2
from scipy.spatial.transform import Rotation
from typing_extensions import Concatenate, ParamSpec

from cylindra._dask import compute, delayed
from cylindra.const import Mode, nm
from cylindra.const import PropertyNames as H
from cylindra.utils import angle_corr, map_coordinates_task, rotated_auto_zncc, roundint

if TYPE_CHECKING:
    import polars as pl

    from cylindra.components._peak import FTPeakInfo
    from cylindra.components.spline import CylSpline
    from cylindra.components.tomogram._cyl_tomo import CylTomogram

_P = ParamSpec("_P")
_R = TypeVar("_R")


class BatchCallable(Protocol[_P, _R]):
    """
    Protocol for batch process decorator.

    This protocol enables static type checking of methods decorated with
    `@batch_process`.
    """

    @overload
    def __call__(
        self, i: Literal[None], *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_R]:
        ...

    @overload
    def __call__(self, i: int, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        ...

    @overload
    def __call__(
        self, i: Iterable[int] | None, *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_R]:
        ...

    def __call__(self, i, *args, **kwargs):
        ...


def batch_process(
    func: Callable[Concatenate[CylTomogram, Any, _P], _R]
) -> BatchCallable[_P, _R]:
    """Enable running function for every splines."""

    @wraps(func)
    def _func(self: CylTomogram, i=None, **kwargs):
        if isinstance(i, int):
            out = func(self, i=i, **kwargs)
            return out

        # Determine along which spline function will be executed.
        if i is None:
            i_list = range(len(self.splines))
        elif not hasattr(i, "__iter__"):
            raise TypeError("'i' must be int or iterable of int if specified")
        else:
            i_list = []
            for i_ in i:
                if -len(self.splines) <= i_ < 0:
                    i_list.append(i_ + len(self.splines))
                elif 0 <= i_ < len(self.splines):
                    i_list.append(i_)
                else:
                    raise ValueError(f"Index {i_} is out of bound")

            if len(i_list) > len(set(i_list)):
                raise ValueError("Indices cannot contain duplicated values.")

        # Run function along each spline
        out = []
        for i_ in i_list:
            try:
                result = func(self, i=i_, **kwargs)
            except Exception as e:
                e.args = (f"{e} (Exception at spline-{i_})",)
                raise e
            else:
                out.append(result)

        return out

    return _func


class FitResult:
    def __init__(self, residual: NDArray[np.float64]):
        self.residual = residual

    @property
    def rmsd(self) -> float:
        """Root mean square deviation."""
        return np.sqrt(np.sum(self.residual**2) / self.residual.shape[0])


class ImageWithPeak:
    def __init__(self, image: ip.ImgArray, power: ip.ImgArray, peaks: list[FTPeakInfo]):
        self.image = image
        self.power = power
        self.peaks = peaks


def dask_angle_corr(
    imgs, ang_centers, drot: float = 7, nrots: int = 29
) -> NDArray[np.float32]:
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers, strict=True):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks)[0]


def prepare_radii(
    spl: CylSpline, radius: nm | Literal["local", "global"]
) -> Sequence[float]:
    if isinstance(radius, str):
        if radius == "global":
            if spl.radius is None:
                raise ValueError("Global radius is not measured yet.")
            radii = np.full(spl.anchors.size, spl.radius, dtype=np.float32)
        elif radius == "local":
            if not spl.props.has_loc(H.radius):
                raise ValueError("Local radii is not measured yet.")
            radii = spl.props.loc[H.radius].to_numpy()
        else:
            raise ValueError("`radius` must be 'local' or 'global' if string.")
    else:
        if radius <= 0:
            raise ValueError("`radius` must be a positive float.")
        radii = np.full(spl.anchors.size, radius, dtype=np.float32)
    return radii


def angle_uniform_filter(input, size, mode=Mode.mirror, cval=0):
    """Uniform filter of angles."""
    phase = np.exp(1j * input)
    out = ndi.convolve1d(phase, np.ones(size), mode=mode, cval=cval)
    return np.angle(out)


def soft_mask_edges(
    subtomograms: ip.ImgArray,
    spl: CylSpline,
    anc: NDArray[np.float32],
    scale: float,
    edge_sigma: float | None,
) -> ip.ImgArray:
    if edge_sigma is None:
        return subtomograms
    # mask XY-region outside the cylinders with sigmoid function.
    yy, xx = np.indices(subtomograms.sizesof("yx"))
    yc, xc = np.array(subtomograms.sizesof("yx")) / 2 - 0.5
    yr = yy - yc
    xr = xx - xc
    for _j, ds in enumerate(spl.map(anc, der=1)):
        _, vy, vx = ds
        distance: NDArray[np.float64] = (
            np.abs(-xr * vy + yr * vx) / np.sqrt(vx**2 + vy**2) * scale
        )
        distance_cutoff = spl.config.fit_width / 2
        if edge_sigma == 0:
            mask_yx = (distance > distance_cutoff).astype(np.float32)
        else:
            mask_yx = 1 / (1 + np.exp((distance - distance_cutoff) / edge_sigma))
        mask = np.stack([mask_yx] * subtomograms.shape.z, axis=0)
        subtomograms[_j] *= mask
    return subtomograms


def shift_coords(
    coords_px: NDArray[np.float32],
    shifts: NDArray[np.float32],
    refined_tilt_rad: NDArray[np.float32],
) -> NDArray[np.float32]:
    shifts_3d = np.stack(
        [shifts[:, 0], np.zeros(shifts.shape[0]), shifts[:, 1]], axis=1
    )
    rotvec = np.zeros(shifts_3d.shape, dtype=np.float32)
    rotvec[:, 0] = -refined_tilt_rad
    rot = Rotation.from_rotvec(rotvec)
    return coords_px + rot.apply(shifts_3d)


def prep_loader_for_refine(
    self: CylTomogram,
    spl: CylSpline,
    ancs: NDArray[np.float32],
    binsize: int,
    twists: NDArray[np.float32] | None = None,
) -> SubtomogramLoader:
    input_img = self._get_multiscale_or_original(binsize)

    depth_px = self.nm2pixel(spl.config.fit_depth, binsize=binsize)
    width_px = self.nm2pixel(spl.config.fit_width, binsize=binsize)

    if twists is None:
        rotation = None
    else:
        rotation = -np.deg2rad(twists)
    mole = spl.anchors_to_molecules(ancs, rotation=rotation)
    if binsize > 1:
        mole = mole.translate(-self.multiscale_translation(binsize))
    scale = input_img.scale.x

    # Load subtomograms rotated by twisting. All the subtomograms should look similar.
    arr = input_img.value
    loader = SubtomogramLoader(
        arr,
        mole,
        order=1,
        scale=scale,
        output_shape=(width_px, depth_px, width_px),
        corner_safe=True,
    )
    return loader


def get_twists(
    length: float,
    nancs: int,
    space: float,
    twist: float,
    npf: int,
):
    """Twist angles for each anchor point for spline refinement."""
    twist_interv = length / (nancs - 1)
    twists = np.arange(nancs) * twist_interv / space * twist
    pf_ang = 360 / npf
    twists %= pf_ang
    twists[twists > pf_ang / 2] -= pf_ang
    return twists


@delayed
def delayed_translate(img: ip.ImgArray, shift) -> ip.ImgArray:
    return img.affine(translation=shift, mode=Mode.constant, cval=0)


@delayed
def delayed_zncc_maximum(
    img: ip.ImgArray,
    tmp: ip.ImgArray,
    max_shifts: int,
    twist: float,
):
    shift = -ip.zncc_maximum(tmp, img, max_shifts=max_shifts)
    rad = np.deg2rad(twist)
    cos, sin = np.cos(rad), np.sin(rad)
    zxrot = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
    return shift @ zxrot


def mask_missing_wedge(
    img: ip.ImgArray,
    tilt_model: TiltSeriesModel,
    quat: NDArray[np.float32],
) -> ip.ImgArray:
    """Mask the missing wedge of the image and return the real image."""
    shape = (img.shape[0], 1, img.shape[1])
    # central slice theorem
    mask3d = tilt_model.create_mask(Rotation(quat), shape)
    mask = mask3d[:, 0, :]
    return ip.asarray(ifft2(fft2(img.value) * mask).real, like=img)


def get_thickness(spl: CylSpline, scale: nm) -> tuple[nm, nm]:
    thick_inner_px = spl.config.thickness_inner / scale
    thick_outer_px = spl.config.thickness_outer / scale
    return thick_inner_px, thick_outer_px


def get_radial_prof(
    input_img: ip.ImgArray | ip.LazyImgArray,
    spl: CylSpline,
    anc: float,
    r_range: tuple[nm, nm],
    depth: nm,
) -> da.Array[np.float32]:
    coords = spl.local_cylindrical(r_range, depth, anc, scale=input_img.scale.x)
    task = map_coordinates_task(input_img, coords)
    prof = da.mean(
        da.from_delayed(task, shape=coords.shape[1:], dtype=np.float32), axis=(1, 2)
    )
    return prof


def get_radius_offset(min_radius_px, max_radius_px) -> nm:
    n_radius = roundint(max_radius_px - min_radius_px)
    return (min_radius_px + max_radius_px - n_radius + 1) / 2


@delayed
def _lazy_rotated_auto_zncc(img, degrees, max_shifts):
    return rotated_auto_zncc(img, degrees, max_shifts=max_shifts)


def multi_rotated_auto_zncc(
    imgs: ip.ImgArray, degrees: NDArray[np.float32], max_shift_px: int
) -> NDArray[np.float32]:
    tasks = [_lazy_rotated_auto_zncc(subimg, degrees, max_shift_px) for subimg in imgs]
    return np.stack(compute(*tasks), axis=0)


def normalize_chunk_length(img, chunk_length: nm | None) -> nm:
    if chunk_length is None:
        if isinstance(img, ip.LazyImgArray):
            chunk_length = 72.0
        else:
            chunk_length = 999999
    return chunk_length


def get_component(df: pl.DataFrame, key: str, idx: int, default=None):
    try:
        return df[key][idx]
    except (KeyError, IndexError):
        return default
