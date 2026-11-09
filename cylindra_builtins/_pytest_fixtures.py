import sys
from contextlib import suppress

import pytest


@pytest.fixture
def ui(make_napari_viewer, request: "pytest.FixtureRequest"):
    import napari

    from cylindra.core import ACTIVE_WIDGETS, _delete_instance, start
    from cylindra.widget_utils import find_dock_widget
    from cylindra.widgets.sta import StaParameters

    viewer: napari.Viewer = make_napari_viewer()
    _ui = start(viewer=viewer)
    if request.config.getoption("--show-viewer", default=None):
        viewer.show()
    yield _ui

    _wait_for_thread_workers()
    cache_roots = [_ui, *ACTIVE_WIDGETS]
    if _ui._batch is not None:
        cache_roots.append(_ui._batch)
    _ui._disconnect_layerlist_events()
    dock_widgets = [find_dock_widget(w) for w in viewer.window.dock_widgets.values()]
    for dock in dock_widgets:
        if dock:
            dock.close()
    for _w in ACTIVE_WIDGETS:
        with suppress(RuntimeError):
            _w.close()
    # Widgets of this test must not be touched in later tests. Once released, their
    # C++ objects may be deleted (e.g. along with the parent) while the Python objects
    # are still waiting for the garbage collection.
    ACTIVE_WIDGETS.clear()
    if batch := _ui._batch:
        with suppress(RuntimeError):
            batch.constructor.close()
            batch.close()
    del _ui.tomogram._image
    _ui.close()
    _delete_instance()
    if sv := StaParameters._viewer:
        with suppress(RuntimeError):
            sv.close()
        StaParameters._viewer = None
    viewer.close()
    _clear_console_namespace(_ui, viewer)
    _release_magicclass_references(cache_roots)


def _wait_for_thread_workers(timeout_ms: int = 5000) -> None:
    """Wait for the running thread workers and process their pending callbacks.

    Signals of workers keep references to the callbacks (and thus to the GUI) from the
    C++ side, so workers must be finished before releasing them.
    """
    from qtpy.QtCore import QThreadPool
    from qtpy.QtWidgets import QApplication

    QThreadPool.globalInstance().waitForDone(timeout_ms)
    for _ in range(3):
        QApplication.processEvents()


def _clear_console_namespace(*objs) -> None:
    """Remove `objs` from the namespace of the in-process IPython kernel (singleton)."""
    shell_mod = sys.modules.get("ipykernel.inprocess.ipkernel")
    if shell_mod is None or not shell_mod.InProcessInteractiveShell.initialized():
        return
    ns = shell_mod.InProcessInteractiveShell.instance().user_ns
    ids = {id(obj) for obj in objs}
    for key in [k for k, v in ns.items() if id(v) in ids]:
        del ns[key]


def _release_magicclass_references(roots: list) -> None:
    """Release the references to `roots` (and their descendants) held by magicclass.

    magicclass descriptors (fields, thread workers, etc.) are class attributes, and
    they cache the objects created for each GUI instance in dicts keyed by
    `id(instance)`. Similarly, `connect_async` keeps the last worker, whose signals
    refer to the GUI via the connected callbacks. These references keep every GUI
    created in a test alive until the end of the session, which consumes a lot of
    memory (~100 MB per test) and slows down the tests in environments with small
    memory such as the macOS runners of GitHub Actions. This is a workaround until
    magicclass itself is fixed.
    """
    import gc

    from magicclass._gui._base import MagicTemplate
    from magicclass._gui._macro import PropertyGroup
    from magicclass.fields import FieldGroup, MagicField
    from magicclass.fields._fields import AsyncConnection
    from magicclass.utils import thread_worker

    cache_attrs: list[tuple[type, tuple[str, ...]]] = [
        (MagicField, ("_guis",)),
        (thread_worker, ("_objects", "_progressbars")),
        (FieldGroup, ("_containers",)),
        (PropertyGroup, ("_instances",)),
    ]
    attrs_of_type: dict[type, tuple[str, ...]] = {}
    caches: list[dict[int, object]] = []
    for obj in gc.get_objects():
        obj_type = type(obj)
        if obj_type is AsyncConnection:
            worker = obj.running_worker
            if worker is not None and not getattr(worker, "_running", False):
                obj.running_worker = None
            continue
        if (attrs := attrs_of_type.get(obj_type)) is None:
            attrs = attrs_of_type[obj_type] = next(
                (a for cls, a in cache_attrs if issubclass(obj_type, cls)), ()
            )
        if attrs:
            ns = getattr(obj, "__dict__", {})
            caches.extend(c for a in attrs if isinstance(c := ns.get(a), dict))
    if (palette := sys.modules.get("magicclass.command_palette")) is not None:
        caches.append(palette._PALETTES)

    # Objects cached for a GUI may also be keys of other caches (e.g. a child
    # magicclass created by a field), so collect all of them recursively.
    ids: set[int] = set()
    to_visit = list(roots)
    while to_visit:
        obj = to_visit.pop()
        if id(obj) in ids:
            continue
        ids.add(id(obj))
        if isinstance(obj, MagicTemplate):
            to_visit.extend(obj.__magicclass_children__)
            # dynamically appended children (such as plugin menus) are only in `_list`
            to_visit.extend(getattr(obj, "_list", None) or ())
            # every magicclass has its own macro object, even if it is a child
            if (macro := getattr(obj, "_macro_instance", None)) is not None:
                to_visit.append(macro)
        for cache in caches:
            if (cached := cache.get(id(obj))) is not None:
                to_visit.append(cached)
    for cache in caches:
        for _id in ids.intersection(cache.keys()):
            del cache[_id]


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
        with suppress(RuntimeError):
            widget.close()
    ACTIVE_WIDGETS.clear()
    use_app().process_events()
