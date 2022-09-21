from __future__ import annotations
import numpy as np
import impy as ip
from dask import array as da
from acryo import SubtomogramLoader
from ..const import Mode

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