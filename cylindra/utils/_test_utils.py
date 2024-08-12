from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cylindra.components import CylSpline
from cylindra.types import MoleculesLayer
from cylindra.widgets.sta import StaParameters


def assert_molecules_equal(
    mole1: MoleculesLayer, mole2: MoleculesLayer, rtol=1e-5, atol=1e-2
):
    assert_allclose(mole1.molecules.pos, mole2.molecules.pos, rtol=rtol, atol=atol)
    # comparing quaternion is not safe.
    assert_allclose(mole1.molecules.x, mole2.molecules.x, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.y, mole2.molecules.y, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.z, mole2.molecules.z, rtol=rtol, atol=atol)


def assert_splines_close(spl0: CylSpline, spl1: CylSpline, tol=1e-2):
    diff = np.sqrt(np.sum((spl0.partition(n=100) - spl1.partition(n=100)) ** 2))
    assert diff < tol


TEST_DIR = Path(__file__).parent.parent.parent / "tests"
PROJECT_DIR_13PF = TEST_DIR / "test_project_13pf"
PROJECT_DIR_14PF = TEST_DIR / "test_project_14pf"


@pytest.fixture
def ui(make_napari_viewer, request: pytest.FixtureRequest):
    import napari

    from cylindra.core import ACTIVE_WIDGETS, start

    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    if request.config.getoption("--show-viewer"):
        viewer.show()
    yield _ui

    _ui._disconnect_layerlist_events()
    for dock in viewer.window._dock_widgets.values():
        dock.close()
    for _w in ACTIVE_WIDGETS:
        with suppress(RuntimeError):
            _w.close()
    if batch := _ui._batch:
        with suppress(RuntimeError):
            batch.constructor.close()
            batch.close()
    del _ui.tomogram._image
    _ui.close()
    if sv := StaParameters._viewer:
        with suppress(RuntimeError):
            sv.close()
        StaParameters._viewer = None
    viewer.close()
