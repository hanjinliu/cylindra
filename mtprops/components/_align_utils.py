from __future__ import annotations
import itertools
from typing import Union, overload
from typing_extensions import Literal
import numpy as np
from numpy.typing import ArrayLike
import impy as ip

from .molecules import from_euler, Molecules


def align_image_to_template(
    image_avg: ip.ImgArray,
    template: ip.ImgArray,
    mask: ip.ImgArray = None,
    max_shifts = None,
) -> tuple[float, np.ndarray]:
    if image_avg.shape != template.shape:
        raise ValueError(
            f"Shape mismatch. Shape of Average image is {tuple(image_avg.shape)} while "
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
        img_rot = image_avg.rotate(yrot, cval=0, dims="zx")
        img_rot_ft = img_rot.fft()
        shift = ip.ft_pcc_maximum(img_rot_ft, template_ft, max_shifts=max_shifts)
        shifts.append(shift)
        img_rot_shift = img_rot.affine(translation=shift)
        corr = ip.zncc(img_rot_shift*mask, masked_template)
        corrs.append(corr)
    
    iopt = np.argmax(corrs)
    return np.deg2rad(rots[iopt]), shifts[iopt]

RangeLike = tuple[float, int]
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
    from scipy.spatial.transform import Rotation
    shift_corrected = Rotation.from_rotvec(rotvec).apply(shift)
    return molecules.translate_internal(shift_corrected).rotate_by_rotvec_internal(rotvec)