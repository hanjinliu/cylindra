from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
import impy as ip

try:
    from .. import _cpp_ext
except ImportError:
    # In case build failed
    pass

def viterbi(
    score: np.ndarray,
    origin: np.ndarray,
    zvec: np.ndarray,
    yvec: np.ndarray,
    xvec: np.ndarray,
    dist_min: float,
    dist_max: float,
    skew_max: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    One-dimensional Viterbi algorithm for contexted subtomogram alignment.

    Parameters
    ----------
    score : (N, Nz, Ny, Nx) array
        Array of score landscape.
    origin : (N, 3) array
        World coordinates of origin of local coordinates.
    zvec : (N, 3) array
        World coordinate vectors of z-axis.
    yvec : (N, 3) array
        World coordinate vectors of y-axis.
    xvec : (N, 3) array
        World coordinate vectors of x-axis.
    dist_min : float
        Minimum distance between subtomograms.
    dist_max : float
        Maximum distance between subtomograms.
    skew_max : float, optional
        Maximum skew between subtomograms, if given.

    Returns
    -------
    (N, 3) int array and float
        Optimal indices and optimal score.
    """
    if dist_min >= dist_max:
        raise ValueError("'dist_min' must be smaller than 'dist_max'.")
    
    grid = _cpp_ext.ViterbiGrid(score, origin, zvec, yvec, xvec)
    if skew_max is None:
        out = grid.viterbi_simple(dist_min, dist_max)
    else:
        out = grid.viterbi(dist_min, dist_max, skew_max)
    return out


def zncc_landscape(
    img0: np.ndarray,
    img1: np.ndarray,
    max_shifts: tuple[float, float ,float], 
    upsample_factor: int = 10,
):
    lds = ip.zncc_landscape(
        ip.asarray(img0, axes="zyx"),
        ip.asarray(img1, axes="zyx"),
        max_shifts=max_shifts
    )
    
    upsampled_max_shifts = (np.asarray(max_shifts) * upsample_factor).astype(np.int32)
    center = np.array(lds.shape) / 2 - 0.5
    mesh = np.meshgrid(
        *[np.arange(-width, width+1)/upsample_factor + c
          for c, width in zip(center, upsampled_max_shifts)], 
        indexing="ij",
    )
    coords = np.stack(mesh, axis=0)
    
    return ndi.map_coordinates(
        lds, coords, order=3, mode="constant", cval=0., prefilter=True
    )
    