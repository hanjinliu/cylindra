from __future__ import annotations
import itertools
from typing import Iterable, Union, overload
from typing_extensions import Literal
import numpy as np
from numpy.typing import ArrayLike
import impy as ip

from .molecules import from_euler, Molecules
from ..utils import set_gpu


def align_image_to_template(
    image: ip.ImgArray,
    template: ip.ImgArray,
    mask: ip.ImgArray = None,
    max_shifts = None,
) -> tuple[float, np.ndarray]:
    """Determine the shift and rotation that will align image to template."""
    if image.shape != template.shape:
        raise ValueError(
            f"Shape mismatch. Shape of Average image is {tuple(image.shape)} while "
            f"shape of template image is {tuple(template.shape)}"
        )
    if mask is None:
        mask = 1
    corrs: list[float] = []
    shifts: list[np.ndarray] = []
    rots = np.linspace(-15, 15, 7)
    masked_template = template*mask
    template_ft = masked_template.fft()
    for yrot in rots:
        img_rot = image.rotate(yrot, cval=0, dims="zx")
        img_rot_ft = img_rot.fft()
        shift = ip.ft_pcc_maximum(img_rot_ft, template_ft, max_shifts=max_shifts)
        shifts.append(shift)
        img_rot_shift = img_rot.affine(translation=shift)
        corr = ip.zncc(img_rot_shift*mask, masked_template)
        corrs.append(corr)
    
    iopt = np.argmax(corrs)
    return np.deg2rad(rots[iopt]), shifts[iopt]

RangeLike = tuple[float, float]
Ranges = Union[RangeLike, tuple[RangeLike, RangeLike, RangeLike]]

def _normalize_a_range(rng: RangeLike) -> tuple[float, int]:
    if len(rng) != 2:
        raise TypeError("Range must be defined by (float, int).")
    max_rot, step = rng
    return float(max_rot), float(step)
        
def _normalize_ranges(rng: Ranges) -> tuple[tuple[float, int], tuple[float, int], tuple[float, int]]:
    if isinstance(rng, tuple) and isinstance(rng[0], tuple):
        return tuple(_normalize_a_range(r) for r in rng)
    else:
        rng = _normalize_a_range(rng)
        return (rng,) * 3

@overload
def normalize_rotations(rotations: Literal[None]) -> None:
    ...

@overload
def normalize_rotations(rotations: Ranges) -> np.ndarray:
    ...

def normalize_rotations(rotations):
    """
    Normalize various rotation expressions to quaternions.

    Parameters
    ----------
    rotations : tuple of float and int, or list of it, optional
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
        if len(quat) == 1:
            rotations = None
        else:
            rotations = np.stack(quat, axis=0)
        
    return rotations

def transform_molecules(
    molecules: Molecules, 
    shift: ArrayLike, 
    rotvec: ArrayLike, 
) -> Molecules:
    """Shift and rotate molecules around their own coordinate."""
    from scipy.spatial.transform import Rotation
    shift_corrected = Rotation.from_rotvec(rotvec).apply(shift)
    return molecules.translate_internal(shift_corrected).rotate_by_rotvec_internal(rotvec)

def align_subvolume(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_ft: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> np.ndarray:
    with ip.silent(), set_gpu():    
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input_ft = (subvol_filt * mask).fft()
        shift = ip.ft_pcc_maximum(
            input_ft,
            template_ft, 
            upsample_factor=20, 
            max_shifts=max_shift
        )
    return shift

def align_subvolume_list(
    subvol_set: Iterable[ip.ImgArray],
    cutoff: float,
    mask: ip.ImgArray,
    template_ft: ip.ImgArray,
    template_for_zncc: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray]:
    corrs: list[float] = []
    all_shifts: list[np.ndarray] = []
    with ip.silent(), set_gpu():
        for subvol in subvol_set:
            subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
            input_ft = (subvol_filt * mask).fft()
            shift = ip.ft_pcc_maximum(
                input_ft,
                template_ft, 
                upsample_factor=20, 
                max_shifts=max_shift,
            )
            all_shifts.append(shift)
            corr = shifted_zncc(subvol_filt, template_for_zncc, mask, shift)
            corrs.append(corr)
    
    iopt = np.argmax(corrs)
    return iopt, all_shifts[iopt]

def align_subvolume_multitemplates(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_ft_list: list[ip.ImgArray],
    template_for_zncc_list: list[ip.ImgArray],
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray]:
    corrs: list[float] = []
    all_shifts: list[np.ndarray] = []
    with ip.silent(), set_gpu():
        subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
        input_ft = (subvol_filt * mask).fft()
        for tmp_ft, tmp_zncc in zip(template_ft_list, template_for_zncc_list):
            shift = ip.ft_pcc_maximum(
                input_ft,
                tmp_ft, 
                upsample_factor=20, 
                max_shifts=max_shift,
            )
            all_shifts.append(shift)
            corr = shifted_zncc(subvol_filt, tmp_zncc, mask, shift)
            corrs.append(corr)
    
    iopt = np.argmax(corrs)
    return iopt, all_shifts[iopt]


def align_subvolume_list_multitemplates(
    subvol_set: Iterable[ip.ImgArray],
    cutoff: float,
    mask: ip.ImgArray,
    template_ft_list: list[ip.ImgArray],
    template_for_zncc_list: list[ip.ImgArray],
    max_shift: tuple[int, int, int],
) -> tuple[tuple[int, int], np.ndarray]:
    corrs: list[list[float]] = []  # corrs[i, j] := i-th rotation, j-th template
    all_shifts: list[np.ndarray] = []
    with ip.silent(), set_gpu():
        for subvol in subvol_set:
            subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
            input_ft = (subvol_filt * mask).fft()
            current_corrs: list[float] = []
            current_all_shifts: list[np.ndarray] = []
            for tmp_ft, tmp_zncc in zip(template_ft_list, template_for_zncc_list):
                shift = ip.ft_pcc_maximum(
                    input_ft,
                    tmp_ft, 
                    upsample_factor=20, 
                    max_shifts=max_shift,
                )
                all_shifts.append(shift)
                corr = shifted_zncc(subvol_filt, tmp_zncc, mask, shift)
                current_corrs.append(corr)
            
            corrs.append(current_corrs)
            all_shifts.append(current_all_shifts)
    
    _corrs = np.array(corrs)
    iopt, jopt = np.unravel_index(np.argmax(_corrs), _corrs.shape)
    return (iopt, jopt), all_shifts[iopt][jopt]

def shifted_zncc(
    subvol: ip.ImgArray,
    ref: ip.ImgArray,
    mask: ip.ImgArray,
    shift: np.ndarray
) -> float:
    shifted_subvol = subvol.affine(translation=shift)
    return ip.zncc(shifted_subvol*mask, ref)