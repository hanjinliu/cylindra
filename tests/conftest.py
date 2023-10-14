import pytest
from contextlib import suppress
from cylindra.widgets.sta import StaParameters


@pytest.fixture
def ui(make_napari_viewer):
    from cylindra.core import start, _ACTIVE_WIDGETS
    import napari

    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    viewer.show()
    yield _ui

    _ui._disconnect_layerlist_events()
    for _w in _ui._active_widgets:
        with suppress(RuntimeError):
            _w.close()
    for _w in _ACTIVE_WIDGETS:
        with suppress(RuntimeError):
            _w.close()
    if batch := _ui._batch:
        with suppress(RuntimeError):
            batch.constructor.close()
            batch.close()
            # batch.constructor.native.deleteLater()
    _ui.close()
    if sv := StaParameters._viewer:
        with suppress(RuntimeError):
            sv.close()
        StaParameters._viewer = None
    viewer.close()
