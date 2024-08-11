from typing import Annotated

import numpy as np
import polars as pl

from cylindra._dask import compute
from cylindra.plugin.core import register_function
from cylindra.utils import map_coordinates_task
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets._annotated import SplinesType


@register_function
def intensity_statistics_along_spline(
    ui: CylindraMainWidget,
    spline: SplinesType,
    shape: tuple[float, float] = (5.0, 5.0),
    depth: float = 5.0,
):
    """
    Calculate mean, std, max, and min intensity along a spline.

    Parameters
    ----------
    ui : CylindraMainWidget
        The main widget.
    spline : SplinesType
        The spline ID to calculate statistics along.
    shape : tuple[float, float], default (5.0, 5.0)
        The shape (in nm) of the local region to calculate statistics.
    depth : float, default 5.0
        The depth (in nm) of the local region to calculate statistics.
    """
    for i in spline:
        spl = ui.splines[i]
        coords = spl.local_cartesian(shape, depth, scale=ui.tomogram.scale)
        tasks = [
            map_coordinates_task(ui.tomogram.image, each_coords, order=3)
            for each_coords in coords
        ]
        out = np.stack(compute(*tasks), axis=0)
        df = pl.DataFrame(
            {
                "intensity_mean": np.mean(out, axis=(1, 2, 3)),
                "intensity_std": np.std(out, axis=(1, 2, 3)),
                "intensity_max": np.max(out, axis=(1, 2, 3)),
                "intensity_min": np.min(out, axis=(1, 2, 3)),
            }
        )
        spl.props.update_loc(df, window_size=depth)
    return None


@register_function
def calculate_spline_tilt(
    ui: CylindraMainWidget,
    spline: SplinesType,
    tilt_axis_degree: Annotated[float, {"max": 90.0}] = 0.0,
):
    for i in spline:
        spl = ui.splines[i]
        dr = spl.map(der=1)
        radians = np.arctan2(dr[:, 2], dr[:, 1])
        degrees = np.abs(np.rad2deg(radians) - tilt_axis_degree)
        over_90 = degrees > 90
        degrees[over_90] = 180 - degrees[over_90]
        spl.props.update_loc(pl.DataFrame({"tilt_yx": degrees}), window_size=0.0)
        spl.props.update_glob({"tilt_yx": np.mean(degrees)})
    return None
