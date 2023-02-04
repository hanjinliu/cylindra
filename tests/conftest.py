import pytest
from cylindra import start
import napari

@pytest.fixture
def ui(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    viewer.show()
    yield _ui
    
    _ui.parent_viewer.layers.events.removing.disconnect()
    _ui.parent_viewer.layers.events.removed.disconnect()
    _ui.close()
    if sv := _ui.sta.sub_viewer:
        sv.close()
    viewer.close()
