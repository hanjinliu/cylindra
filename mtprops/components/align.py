

from __future__ import annotations
import itertools
from typing import Iterable, Union, NamedTuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
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


class AlignmentResult(NamedTuple):
    """The optimal alignment result."""
    
    label: int | tuple[int, int]
    shift: np.ndarray
    quat: np.ndarray
    corr: float


class AlignmentModel:
    """A helper class to describe an alignment model."""
    
    def __init__(
        self,
        template: ip.ImgArray,
        mask: ip.ImgArray | None = None,
        cutoff: float = 0.5,
        rotations: Ranges | None = None,
        method: str = "pcc",
    ):
        self._template = template
        self.cutoff = cutoff
        if mask is None:
            self.mask = 1
        else:
            if template.shape != mask.shape:
                raise ValueError("Shape mismatch between tempalte image and mask image.")
            self.mask = mask
        self.quaternions = normalize_rotations(rotations)
        self.method = method.lower()
        self.align_func = self._get_alignment_function()
        self.template_input = self._get_template_input()
    
    def align(
        self, 
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float]
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : ip.ImgArray
            Subvolume to be aligned
        max_shifts : tuple[float, float, float]
            Maximum shift in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        iopt, shift, corr = self.align_func(
            img, self.cutoff, self.mask, self.template_input, max_shifts
        )
        if isinstance(iopt, int):
            quat = self.quaternions[iopt]
        else:
            quat = self.quaternions[iopt[0]]
        return AlignmentResult(label=iopt, shift=shift, quat=quat, corr=corr)
    
    def fit(
        self, 
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float],
        cval: float = None,
    ) -> tuple[ip.ImgArray, AlignmentResult]:
        result = self.align(img, max_shifts=max_shifts)
        rotator = Rotation.from_quat(result.quat)
        matrix = _compose_rotation_matrices(img.shape, [rotator])[0]
        if cval is None:
            cval = np.percentile(img, 1)
        img_trans = (
            img
            .affine(translation=result.shift, cval=cval)
            .affine(matrix=matrix, cval=cval)
        )
        return img_trans, result
    
    @property
    def is_multi_templates(self) -> bool:
        """
        Whether alignment parameters requires multi-templates.
        "Multi-template" includes alignment with subvolume rotation.
        """
        return self.has_rotation or self._template.ndim == 4
    
    @property
    def is_single_template(self) -> bool:
        return not self.is_multi_templates
    
    @property
    def has_rotation(self) -> bool:
        return self.quaternions.shape[0] > 1
    
    def _get_rotators(self, inv: bool = False) -> list[Rotation]:
        if inv:
            return [Rotation.from_quat(r).inv() for r in self.quaternions]
        else:
            return [Rotation.from_quat(r) for r in self.quaternions]
    
    def _get_template_input(self) -> ip.ImgArray:
        """
        Returns proper template image for alignment.
        
        Template dimensionality will be dispatched according to the input parameters.
        Returned template should be used in line of the :func:`get_alignment_function`.

        Returns
        -------
        ip.ImgArray
            Template image(s). Its axes varies depending on the input.
            - no rotation, single template image ... "zyx"
            - has rotation, single template image ... "rzyx"
            - no rotation, many template images ... "pzyx"
            - has rotation, many template images ... "rpzyx"
        """
        template_input = self._template.lowpass_filter(
            cutoff=self.cutoff, dims="zyx"
        ) * self.mask
        if self.is_multi_templates:
            rotators = self._get_rotators(inv=True)
            matrices = _compose_rotation_matrices(template_input.sizesof("zyx"), rotators)
            cval = np.percentile(template_input, 1)
            template_input: ip.ImgArray = np.stack(
                [template_input.affine(mat, cval=cval) for mat in matrices], axis="r"
            )
        if self.method == "pcc":
            template_input = template_input.fft(dims="zyx")
        return template_input

    def _get_alignment_function(self):
        if self.method == "pcc":
            if self.is_multi_templates:
                f = align_subvolume_multitemplates_pcc
            else:
                f = align_subvolume_pcc
        elif self.method == "zncc":
            if self.is_multi_templates:
                f = align_subvolume_multitemplates_zncc
            else:
                f = align_subvolume_zncc
        else:
            raise ValueError(f"Unsupported method {self.method}.")
        return f


def _compose_rotation_matrices(
    shape: tuple[int, int, int],
    rotators: list[Rotation],
):
    dz, dy, dx = (np.array(shape) - 1) / 2
    # center to corner
    translation_0 = np.array([[1., 0., 0., dz],
                              [0., 1., 0., dy],
                              [0., 0., 1., dx],
                              [0., 0., 0., 1.]],
                              dtype=np.float32)
    # corner to center
    translation_1 = np.array([[1., 0., 0., -dz],
                              [0., 1., 0., -dy],
                              [0., 0., 1., -dx],
                              [0., 0., 0.,  1.]],
                              dtype=np.float32)
    
    matrices = []
    for rot in rotators:
        e_ = np.eye(4)
        e_[:3, :3] = rot.as_matrix()
        matrices.append(translation_0 @ e_ @ translation_1)
    return matrices


def transform_molecules(
    molecules: Molecules, 
    shift: ArrayLike, 
    rotvec: ArrayLike, 
) -> Molecules:
    """Shift and rotate molecules around their own coordinate."""
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
