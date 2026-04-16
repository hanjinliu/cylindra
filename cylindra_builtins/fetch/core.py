import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from magicclass.utils import thread_worker

from cylindra.const import ImageFilter
from cylindra.plugin import register_function

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

URL_ROOT = "https://github.com/hanjinliu/cylindra/raw/refs/heads/main"
COORDS_13 = [[16.0, 186.5, 27.7], [18.0, 52.2, 71.6]]
COORDS_13_FIT = [
    [18.532, 186.874, 29.393],
    [18.985, 153.323, 39.989],
    [19.080, 119.781, 50.611],
    [18.880, 86.249, 61.260],
    [18.200, 52.726, 71.934],
]
COORDS_14 = [[29.0, 106.2, 37.3], [26.0, 22.1, 64.7]]
COORDS_14_FIT = [
    [24.109, 106.656, 38.184],
    [24.091, 85.346, 44.397],
    [24.072, 64.036, 50.610],
    [23.982, 42.734, 56.850],
    [23.892, 21.432, 63.090],
]


@register_function(name="13-PF microtubule")
@thread_worker
def mt_13pf(
    ui: "CylindraMainWidget",
    filter: ImageFilter = ImageFilter.Lowpass,
    with_spline: Literal["none", "roughly fitted", "fitted"] = "none",
):
    """Fetch and open a 13-protofilament microtubule test image."""
    with _fetch_image(f"{URL_ROOT}/tests/13pf_MT.tif") as path:
        yield from _open(ui, path, filter)
        _register_path(ui, with_spline, COORDS_13, COORDS_13_FIT)


@register_function(name="14-PF microtubule")
@thread_worker
def mt_14pf(
    ui: "CylindraMainWidget",
    filter: ImageFilter = ImageFilter.Lowpass,
    with_spline: Literal["none", "roughly fitted", "fitted"] = "none",
):
    """Fetch and open a 14-protofilament microtubule test image."""
    with _fetch_image(f"{URL_ROOT}/tests/14pf_MT.tif") as path:
        yield from _open(ui, path, filter)
        _register_path(ui, with_spline, COORDS_14, COORDS_14_FIT)


@contextmanager
def _fetch_image(url: str):
    import requests

    response = requests.get(url)
    response.raise_for_status()
    filename = url.split("/")[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = f"{tmpdir}/{filename}"
        with open(tmp_file, "wb") as f:
            f.write(response.content)
        yield tmp_file


def _open(ui: "CylindraMainWidget", path, filter):
    yield from ui.open_image.arun(path, tilt_range=(-60, 60), eager=True, filter=filter)


def _register_path(ui: "CylindraMainWidget", with_spline, roughly_fitted, fitted):
    if with_spline == "roughly fitted":
        ui.register_path(roughly_fitted)
    elif with_spline == "fitted":
        ui.register_path(fitted)
