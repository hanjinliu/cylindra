from __future__ import annotations
import itertools
from typing import Iterable, Union
import numpy as np
from numpy.typing import ArrayLike
import impy as ip

from .molecules import from_euler, Molecules
from ..utils import set_gpu


RangeLike = tuple[float, float]
Ranges = Union[RangeLike, tuple[RangeLike, RangeLike, RangeLike]]


def _normalize_a_range(rng: RangeLike) -> RangeLike:
    if len(rng) != 2:
        raise TypeError("Range must be defined by (float, float).")
    max_rot, drot = rng
    return float(max_rot), float(drot)
        
def _normalize_ranges(rng: Ranges) -> Ranges:
    if isinstance(rng, tuple) and isinstance(rng[0], tuple):
        return tuple(_normalize_a_range(r) for r in rng)
    else:
        rng = _normalize_a_range(rng)
        return (rng,) * 3


def normalize_rotations(rotations: Ranges | None) -> np.ndarray:
    """
    Normalize various rotation expressions to quaternions.

    Parameters
    ----------
    rotations : tuple of float and float, or list of it, optional
        Rotation around each axis.

    Returns
    -------
    np.ndarray
        Corresponding quaternions in shape (N, 4).
    """
    if rotations is not None:
        rotations = _normalize_ranges(rotations)
        angles = []
        for max_rot, step in rotations:
            if step == 0:
                angles.append(np.zeros(1))
            else:
                n = int(max_rot / step)
                angles.append(np.linspace(-n*step, n*step, 2*n + 1))
        
        quat: list[np.ndarray] = []
        for angs in itertools.product(*angles):
            quat.append(from_euler(np.array(angs), "zyx", degrees=True).as_quat())
        rotations = np.stack(quat, axis=0)
    else:
        rotations = np.array([[0., 0., 0., 1.]])
        
    return rotations

def transform_molecules(
    molecules: Molecules, 
    shift: ArrayLike, 
    rotvec: ArrayLike, 
) -> Molecules:
    """Shift and rotate molecules around their own coordinate."""
    from scipy.spatial.transform import Rotation
    shift_corrected = Rotation.from_rotvec(rotvec).apply(shift)
    return (
        molecules
        .translate_internal(shift_corrected)
        .rotate_by_rotvec_internal(rotvec)
    )

def transform_molecules_inv(
    molecules: Molecules, 
    shift: ArrayLike, 
    rotvec: ArrayLike, 
):
    """Shift and rotate molecules around their own coordinate. Inverse mapping."""
    from scipy.spatial.transform import Rotation
    shift_corrected = Rotation.from_rotvec(rotvec).apply(shift, inverse=True)
    return (
        molecules
        .rotate_by_rotvec_internal(-rotvec)
        .translate_internal(shift_corrected)
    )

def get_alignment_function(
    method: str = "pcc",
    multi_template: bool = False,
):
    method = method.lower()
    if method == "pcc":
        if multi_template:
            f = align_subvolume_multitemplates_pcc
        else:
            f = align_subvolume_pcc
    elif method == "zncc":
        if multi_template:
            f = align_subvolume_multitemplates_zncc
        else:
            f = align_subvolume_zncc
    else:
        raise ValueError(f"Unsupported method {method}.")
    return f

def align_subvolume_zncc(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray, float]:
    with ip.silent():    
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input = subvol_filt * mask
        shift, zncc = ip.zncc_maximum_with_corr(
            input,
            template, 
            upsample_factor=20, 
            max_shifts=max_shift
        )
    return 0, shift, zncc

def align_subvolume_pcc(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_ft: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> tuple[np.ndarray, float]:
    with ip.silent():    
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input = subvol_filt * mask
        shift, pcc = ip.ft_pcc_maximum_with_corr(
            input.fft(),
            template_ft, 
            upsample_factor=20, 
            max_shifts=max_shift
        )
    return 0, shift, pcc

def align_subvolume_multitemplates_pcc(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_ft_list: list[ip.ImgArray],
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray, float]:
    all_shifts: list[np.ndarray] = []
    all_pcc: list[float] = []
    with ip.silent(), set_gpu():
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input = subvol_filt * mask
        for template_ft in template_ft_list:
            shift, pcc = ip.ft_pcc_maximum_with_corr(
                input.fft(),
                template_ft, 
                upsample_factor=20, 
                max_shifts=max_shift,
            )
            all_shifts.append(shift)
            all_pcc.append(pcc)
    
    iopt = int(np.argmax(all_pcc))
    return iopt, all_shifts[iopt], all_pcc[iopt]

def align_subvolume_multitemplates_zncc(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_list: list[ip.ImgArray],
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray, float]:
    all_shifts: list[np.ndarray] = []
    all_zncc: list[float] = []
    with ip.silent(), set_gpu():
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input = subvol_filt * mask
        for template in template_list:
            shift, zncc = ip.zncc_maximum_with_corr(
                input,
                template, 
                upsample_factor=20, 
                max_shifts=max_shift,
            )
            all_shifts.append(shift)
            all_zncc.append(zncc)
    
    iopt = int(np.argmax(all_zncc))
    return iopt, all_shifts[iopt], all_zncc[iopt]


def align_subvolume_list_multitemplates(
    subvol_set: Iterable[ip.ImgArray],
    cutoff: float,
    mask: ip.ImgArray,
    template_list: list[ip.ImgArray],
    max_shift: tuple[int, int, int],
) -> tuple[tuple[int, int], np.ndarray, float]:
    all_shifts: list[np.ndarray] = []
    all_zncc: list[list[float]] = []  # corrs[i, j] := i-th rotation, j-th template
    with ip.silent(), set_gpu():
        for subvol in subvol_set:
            subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
            input = subvol_filt * mask
            current_zncc: list[float] = []
            current_all_shifts: list[np.ndarray] = []
            for template in template_list:
                shift, corr = ip.zncc_maximum_with_corr(
                    input,
                    template, 
                    upsample_factor=20, 
                    max_shifts=max_shift,
                )
                all_shifts.append(shift)
                current_zncc.append(corr)
            
            all_zncc.append(current_zncc)
            all_shifts.append(current_all_shifts)
    
    _corrs = np.array(all_zncc)
    iopt, jopt = np.unravel_index(np.argmax(_corrs), _corrs.shape)
    return (iopt, jopt), all_shifts[iopt][jopt], _corrs[iopt][jopt]

def shifted_zncc(
    subvol: ip.ImgArray,
    ref: ip.ImgArray,
    mask: ip.ImgArray,
    shift: np.ndarray
) -> float:
    shifted_subvol = subvol.affine(translation=shift)
    return ip.zncc(shifted_subvol*mask, ref)