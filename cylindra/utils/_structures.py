from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import impy as ip
from dask import array as da
from acryo import Molecules, SubtomogramLoader
from cylindra.const import Mode, MoleculesHeader as Mole, GlobalVariables as GVar

from ._correlation import mirror_zncc

def try_all_seams(
    loader: SubtomogramLoader,
    npf: int,
    template: ip.ImgArray,
    mask: ip.ImgArray | None = None,
    cutoff: float = 0.5,
) -> tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]:
    """
    Try all the possible seam positions and compare correlations.

    Parameters
    ----------
    loader : SubtomogramLoader
        An aligned ``acryo.SubtomogramLoader`` object.
    npf : int
        Number of protofilament.
    template : ip.ImgArray
        Template image.
    mask : ip.ImgArray, optional
        Mask image.
    cutoff : float, default is 0.5
        Cutoff frequency applied before calculating correlations.

    Returns
    -------
    tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]
        Correlation, average and boolean array correspond to each seam position.
    """    
    corrs: list[float] = []
    labels: list[np.ndarray] = []  # list of boolean arrays
    
    if mask is None:
        mask = 1.
    
    masked_template = (template * mask).lowpass_filter(cutoff=cutoff, dims="zyx")
    _id = np.arange(len(loader.molecules))
    assert _id.size % npf == 0
    
    # prepare all the labels in advance (only takes up ~0.5 MB at most)
    for pf in range(2*npf):
        res = (_id - pf) // npf
        sl = res % 2 == 0
        labels.append(sl)
    
    # here, dask_array is (N, Z, Y, X) array where dask_array[i] is i-th subtomogram.
    dask_array = loader.construct_dask(output_shape=template.shape)
    averaged_images = da.compute([da.mean(dask_array[sl], axis=0) for sl in labels])[0]
    averaged_images = ip.asarray(np.stack(averaged_images, axis=0), axes="pzyx")
    averaged_images.set_scale(zyx=loader.scale)
    
    corrs: list[float] = []
    for avg in averaged_images:
        avg: ip.ImgArray
        corr = ip.zncc((avg * mask).lowpass_filter(cutoff=cutoff), masked_template)
        corrs.append(corr)
        
    return np.array(corrs), averaged_images, labels


def centering(
    img: ip.ImgArray,
    point: np.ndarray,
    angle: float,
    drot: float = 5, 
    nrots: int = 7,
    max_shifts: float | None = None,
):
    """
    Find the center of cylinder using self-correlation.

    Parameters
    ----------
    img : ip.ImgArray
        Target image.
    point : np.ndarray
        Current center of cylinder.
    angle : float
        The central angle of the cylinder.
    drot : float, default is 5
        Deviation of the rotation angle.
    nrots : int, default is 7
        Number of rotations to try.
    max_shifts : float, optional
        Maximum shift in pixel.

    """
    angle_deg2 = angle_corr(img, ang_center=angle, drot=drot, nrots=nrots)
    
    img_next_rot = img.rotate(-angle_deg2, cval=np.mean(img))
    proj = img_next_rot.proj("y")
    shift = mirror_zncc(proj, max_shifts=max_shifts)
    
    shiftz, shiftx = shift/2
    shift = np.array([shiftz, 0, shiftx])
    rad = -np.deg2rad(angle_deg2)
    cos = np.cos(rad)
    sin = np.sin(rad)
    shift = shift @ [[1.,   0.,  0.],
                     [0.,  cos, sin],
                     [0., -sin, cos]]
    point += shift
    return point

def angle_corr(img: ip.ImgArray, ang_center: float = 0, drot: float = 7, nrots: int = 29):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.shape.y/2+2, img_z.shape)
    img_mirror: ip.ImgArray = img_z["x=::-1"]
    angs = np.linspace(ang_center-drot, ang_center+drot, nrots, endpoint=True)
    corrs = []
    f0 = np.sqrt(img_z.power_spectra(dims="yx", zero_norm=True))
    cval = np.mean(img_z)
    for ang in angs:
        img_mirror_rot = img_mirror.rotate(ang*2, mode=Mode.constant, cval=cval)
        f1 = np.sqrt(img_mirror_rot.power_spectra(dims="yx", zero_norm=True))
        corr = ip.zncc(f0, f1, mask)
        corrs.append(corr)
        
    angle = angs[np.argmax(corrs)]
    return angle

def _molecules_to_spline(mole: Molecules):
    """Convert well aligned molecule positions into a spline."""
    from cylindra.components import CylSpline

    spl = CylSpline(degree=GVar.splOrder)
    npf = int(round(mole.features[Mole.pf].max() + 1))
    all_coords = mole.pos.reshape(-1, npf, 3)
    mean_coords = np.mean(all_coords, axis=1)
    spl.fit_coa(mean_coords, min_radius=GVar.minCurvatureRadius)
    return spl

def _reshaped_positions(mole: Molecules) -> NDArray[np.float32]:
    try:
        pf_label = mole.features[Mole.pf]
        pos_list: list[NDArray[np.float32]] = []  # each shape: (y, ndim)
        for pf in range(pf_label.max() + 1):
            pos_list.append(mole.pos[pf_label == pf])
        pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)
        
    except Exception as e:
        raise TypeError(
            f"Reshaping failed. Molecules must be correctly labeled at {Mole.pf!r} "
            f"feature. Original error is\n{type(e).__name__}: {e}"
        ) from e
    return pos

def calc_interval(mole: Molecules, spline_precision: float) -> NDArray[np.float32]:
    spl = _molecules_to_spline(mole)
    pos = _reshaped_positions(mole)
    u = spl.world_to_y(mole.pos, precision=spline_precision)
    spl_vec = spl(u, der=1)
    
    ny, npf, ndim = pos.shape
    
    # equivalent to padding mode "reflect"
    interv_vec = np.diff(pos, axis=0, append=(2*pos[-1] - pos[-2])[np.newaxis])  
    
    vec_len = np.linalg.norm(spl_vec, axis=2)  # length of spline vector
    vec_norm = spl_vec / vec_len[:, np.newaxis]
    vec_norm = vec_norm.reshape(-1, npf, ndim)  # normalized spline vector
    y_interval = np.sum(interv_vec * vec_norm, axis=2)  # inner product
    y_interval[-1] = -1.  # fill invalid values with -1
    
    properties = y_interval.ravel()
    if properties[0] < 0:
        properties = -properties
    
    return properties

def calc_skew(mole: Molecules, spline_precision: float) -> NDArray[np.float32]:
    spl = _molecules_to_spline(mole)
    pos = _reshaped_positions(mole)
    u = spl.world_to_y(mole.pos, precision=spline_precision)
    ny, npf, ndim = pos.shape
    
    spl_pos = spl(u, der=0)
    spl_vec = spl(u, der=1)
    
    mole_to_spl_vec = (spl_pos - mole.pos).reshape(ny, npf, ndim)
    radius = np.linalg.norm(mole_to_spl_vec, axis=2)
    
    # equivalent to padding mode "reflect"
    interv_vec = np.diff(pos, axis=0, append=(2*pos[-1] - pos[-2])[np.newaxis])  
    interv_vec_len = np.linalg.norm(interv_vec, axis=2)
    interv_vec_norm = interv_vec / interv_vec_len[:, :, np.newaxis]

    spl_vec_len = np.linalg.norm(spl_vec, axis=1)
    spl_vec_norm = spl_vec / spl_vec_len[:, np.newaxis]
    spl_vec_norm = spl_vec_norm.reshape(-1, npf, ndim)  
    
    skew_cross = np.cross(interv_vec_norm, spl_vec_norm, axis=2)  # cross product
    inner = np.sum(skew_cross * mole_to_spl_vec, axis=2)
    skew_sin = np.linalg.norm(skew_cross, axis=2) * np.sign(inner)

    skew = np.rad2deg(2 * interv_vec_len * skew_sin / radius)
    return skew.ravel()
