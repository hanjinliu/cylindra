from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
import numpy as np
import impy as ip
from .spline import build_local_cartesian
from .utils import map_coordinates
from .molecules import Molecules
from .utils import no_verbose

if TYPE_CHECKING:
    from dask import array as da


def iter_subtomograms(
    image: np.ndarray | da.core.Array, 
    mole: Molecules,
    size: int | tuple = 64,
    scale: float = 1.0,
) -> Iterator[ip.ImgArray]:
    centers = mole.pos
    ds_list = mole.y
    if isinstance(size, int):
        size = (size,) * 3
    
    with no_verbose():
        for center, ds in zip(centers, ds_list):
            coords = build_local_cartesian(size, ds, center/scale)
            coords = np.moveaxis(coords, -1, 0)
            subvol = ip.asarray(map_coordinates(image, coords, cval=np.mean), axes="zyx")
            subvol.set_scale(xyz=scale)
            yield subvol


def get_subtomograms(
    image: np.ndarray | da.core.Array, 
    mole: Molecules,
    size: int | tuple = 64,
    scale: float = 1.0,
) -> ip.ImgArray:
    images = list(iter_subtomograms(image, mole, size=size, scale=scale))
    return np.stack(images, axis="p")

def align_subtomograms(
    image: np.ndarray | da.core.Array, 
    mole: Molecules,
    template: ip.ImgArray = None, 
    mask: ip.ImgArray = None,
    scale: float = 1.0,
) -> Molecules:
    """
    Align sub-volumes of tomogram to a template image.
    
    Similar to the method defined in PEET package.

    Parameters
    ----------
    image : da.core.Array
        Tomogram image.
    mole : Molecules
        Molecules object that defines subtomogram regions and orientations.
    template : ip.ImgArray, optional
        Template image.
    mask : ip.ImgArray, optional
        Mask image. Images will be multiplied by mask before calculating correlations.
    scale : float, default is 1.0
        Scale of image.

    Returns
    -------
    Molecules
        Molecules object with translated positions.
    """
    if template is None:
        raise NotImplementedError("Template image is needed.")
    if mask is not None:
        mask = 1
    
    shifts: list[np.ndarray] = []
    
    with no_verbose():
        template_ft = (template * mask).fft()
        for subvol in iter_subtomograms(image, mole, size=template.shape, scale=scale):
            shift = ip.ft_pcc_maximum((subvol*mask).fft(), template_ft)
            shifts.append(shift)
        
    out = mole.translate(np.stack(shifts/scale, axis=0))
    
    return out

