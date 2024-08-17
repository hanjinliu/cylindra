from contextlib import suppress

import pytest


@pytest.fixture
def ui(make_napari_viewer, request: "pytest.FixtureRequest"):
    import napari

    from cylindra.core import ACTIVE_WIDGETS, start
    from cylindra.widgets.sta import StaParameters

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