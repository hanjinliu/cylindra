

from __future__ import annotations
import itertools
from typing import Union, NamedTuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
import impy as ip

from .molecules import from_euler, Molecules


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
        if template.ndim not in (3, 4):
            raise ValueError(
                f"Template image must be 3 or 4 dimensional, got {template.ndim} "
                "dimensional image."
            )
        self._template = template
        self._n_templates = 1 if template.ndim == 3 else template.shape[0]
        self.cutoff = cutoff
        if mask is None:
            self.mask = 1
        else:
            if template.sizesof("zyx") != mask.shape:
                raise ValueError(
                    "Shape mismatch in zyx axes between tempalte image "
                    f"({tuple(template.shape)}) and mask image ({tuple(mask.shape)})."
                )
            self.mask = mask
        self.quaternions = normalize_rotations(rotations)
        self._n_rotations = self.quaternions.shape[0]
        self._method = method.lower()
        self._align_func = self._get_alignment_function()
        self._landscape_func = self._get_landscape_function()
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
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        iopt, shift, corr = self._align_func(
            img, self.cutoff, self.mask, self.template_input, max_shifts
        )
        quat = self.quaternions[iopt % self._n_rotations]
        return AlignmentResult(label=iopt, shift=shift, quat=quat, corr=corr)
    
    def fit(
        self, 
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float],
        cval: float = None,
    ) -> tuple[ip.ImgArray, AlignmentResult]:
        """
        Fit image to template based on the alignment model.

        Parameters
        ----------
        img : ip.ImgArray
            Input image that will be transformed.
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.
        cval : float, optional
            Constant value for padding.

        Returns
        -------
        ip.ImgArray, AlignmentResult
            Transformed input image and the alignment result.
        """
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
    
    def landscape(
        self, 
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float]
    ):
        """
        Compute cross-correlation landscape using current alignment parameters.

        Parameters
        ----------
        img : ip.ImgArray
            Subvolume to be aligned
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        ImgArray
            Landscape image.
        """
        return self._landscape_func(
            img, self.cutoff, self.mask, self.template_input, max_shifts
        )

    
    @property
    def is_multi_templates(self) -> bool:
        """
        Whether alignment parameters requires multi-templates.
        "Multi-template" includes alignment with subvolume rotation.
        """
        return self._n_templates > 1
    
    @property
    def is_single(self) -> bool:
        return self._n_templates == 1 and self._n_rotations == 1
    
    @property
    def has_rotation(self) -> bool:
        return self._n_rotations > 1
    
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
            - has rotation, single template image ... "pzyx"
            - no rotation, many template images ... "pzyx"
            - has rotation, many template images ... "pzyx" and when iterated over the
              first axis yielded images will be (rot0, temp0), (rot0, temp1), ...
        """
        template_input = self._template.lowpass_filter(
            cutoff=self.cutoff, dims="zyx"
        ) * self.mask
        if self.has_rotation:
            rotators = self._get_rotators(inv=True)
            matrices = _compose_rotation_matrices(template_input.sizesof("zyx"), rotators)
            cval = np.percentile(template_input, 1)
            if self.is_multi_templates:
                template_input: ip.ImgArray = np.concatenate(
                    [template_input.affine(mat, cval=cval) for mat in matrices], axis="p"
                )
            else:
                template_input: ip.ImgArray = np.stack(
                    [template_input.affine(mat, cval=cval) for mat in matrices], axis="p"
                )
        if self._method == "pcc":
            template_input = template_input.fft(dims="zyx")
        return template_input

    def _get_alignment_function(self):
        if self._method == "pcc":
            if self.is_multi_templates or self.has_rotation:
                f = align_subvolume_multitemplates_pcc
            else:
                f = align_subvolume_pcc
        elif self._method == "zncc":
            if self.is_multi_templates or self.has_rotation:
                f = align_subvolume_multitemplates_zncc
            else:
                f = align_subvolume_zncc
        else:
            raise ValueError(f"Unsupported method {self._method}.")
        return f

    def _get_landscape_function(self):
        if self._method == "pcc":
            f = zncc_landscape
        elif self._method == "zncc":
            f = pcc_landscape
        else:
            raise ValueError(f"Unsupported method {self._method}.")
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

################################################
#   Alignment functions
################################################

def align_subvolume_zncc(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray, float]:
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


################################################
#   Landscape functions
################################################

def zncc_landscape(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template: ip.ImgArray,
    max_shift: tuple[int, int, int],
):
    subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
    input = subvol_filt * mask
    return ip.zncc_landscape(input, template, max_shift)
    
def pcc_landscape(
    subvol: ip.ImgArray,
    cutoff: float,
    mask: ip.ImgArray,
    template_ft: ip.ImgArray,
    max_shift: tuple[int, int, int],
) -> tuple[np.ndarray, float]:
    subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
    input = subvol_filt * mask
    return ip.ft_pcc_landscape(input.fft(), template_ft, max_shift)
        
# def _zncc_opt(quat, subvol: ip.ImgArray, template: ip.ImgArray, shift):
#     rotated = template.affine(rotation=Rotation.from_quat(quat).as_matrix(), translation=shift)
#     _zncc_opt.rotated = rotated
#     return -ip.zncc(subvol, rotated)

# def align_subvolume_opt(
#     subvol: ip.ImgArray,
#     cutoff: float,
#     mask: ip.ImgArray,
#     template: ip.ImgArray,
#     max_shift: tuple[int, int, int],
# ) -> tuple[int, np.ndarray, float]:
#         subvol_filt = subvol.lowpass_filter(cutoff=cutoff)
#         input = subvol_filt * mask
#         shift = [0, 0, 0]
#         quat = [0, 0, 0, 1]
#         for i in range(4):
#             result = minimize(_zncc_opt, quat, args=(input, template, shift))
#             shift, zncc = ip.zncc_maximum_with_corr(
#                 input, _zncc_opt.rotated, upsample_factor=20, max_shifts=max_shift
#             )
#             quat = result.x
    
#     return 0, shift, zncc