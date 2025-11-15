import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING

from cylindra.plugin import register_function

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

URL_ROOT = "https://github.com/hanjinliu/cylindra/raw/refs/heads/main"


@register_function(name="13 PF microtubule")
def mt_13pf(ui: "CylindraMainWidget"):
    with _fetch_image(f"{URL_ROOT}/tests/13pf_MT.tif") as path:
        ui.open_image(path, tilt_range=(-60, 60), eager=True)


@register_function(name="14 PF microtubule")
def mt_14pf(ui: "CylindraMainWidget"):
    with _fetch_image(f"{URL_ROOT}/tests/14pf_MT.tif") as path:
        ui.open_image(path, tilt_range=(-60, 60), eager=True)


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
