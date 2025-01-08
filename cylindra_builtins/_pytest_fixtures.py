import sys
from contextlib import suppress

import pytest


@pytest.fixture
def ui(make_napari_viewer, request: "pytest.FixtureRequest"):
    import napari

    from cylindra.core import ACTIVE_WIDGETS, start
    from cylindra.widgets.sta import StaParameters

    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    if request.config.getoption("--show-viewer", default=None):
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


@pytest.fixture
def run_cli(make_napari_viewer, monkeypatch):
    from magicclass.utils import thread_worker
    from magicgui.application import use_app

    from cylindra.__main__ import main
    from cylindra.cli import set_testing
    from cylindra.core import ACTIVE_WIDGETS

    viewer = make_napari_viewer()
    set_testing(True)
    monkeypatch.setattr("builtins.input", lambda *_: "")

    def _run_cli(*args):
        sys.argv = [str(a) for a in args]
        main(viewer, ignore_sys_exit=True)

    with thread_worker.blocking_mode():
        yield _run_cli

    for widget in ACTIVE_WIDGETS:
        widget.close()
    ACTIVE_WIDGETS.clear()
    use_app().process_events()
