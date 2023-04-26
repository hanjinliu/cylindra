from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
import impy as ip

try:
    from .._cylindra_ext import ViterbiGrid
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
    grid = ViterbiGrid(score, origin, zvec, yvec, xvec)
    if skew_max is None:
        out = grid.viterbi(dist_min, dist_max)
    else:
        out = grid.viterbi(dist_min, dist_max, skew_max)
    return out
