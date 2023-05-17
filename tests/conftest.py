import pytest


@pytest.fixture
def ui(make_napari_viewer):
    from cylindra import start
    import napari

    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    viewer.show()
    yield _ui

    _ui._disconnect_layerlist_events()
    for _w in _ui._active_widgets:
        _w.close()
    _ui.close()
    if sv := _ui.sta.sub_viewer:
        sv.close()
    if batch := _ui._batch:
        batch.close()
    viewer.close()
