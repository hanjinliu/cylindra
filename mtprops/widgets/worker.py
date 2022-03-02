"""Widgets and functions that deal with napari's workers."""

from __future__ import annotations
import sys
from typing import Union, Callable, TYPE_CHECKING
if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec
from functools import wraps

from napari.qt.threading import GeneratorWorker, FunctionWorker

from magicclass import magicclass, vfield, MagicTemplate
from magicclass.gui._message_box import QtErrorMessageBox

if TYPE_CHECKING:
    from .main import MTPropsWidget

Worker = Union[FunctionWorker, GeneratorWorker]

_P = ParamSpec("_P")

def dispatch_worker(f: Callable[_P, Worker]) -> Callable[_P, None]:
    """
    Open a progress bar and start worker in a parallel thread if function is called from GUI.
    Otherwise (if function is called from script), the worker will be executed as if the 
    function is directly called.
    """
    @wraps(f)
    def wrapper(self: "MTPropsWidget", *args, **kwargs):
        worker: Worker = f(self, *args, **kwargs)
        if self[f.__name__].running:
            self._connect_worker(worker)
        else:
            # run_worker_function(worker)
            worker.run()
        return None
    return wrapper


@magicclass(layout="horizontal", labels=False, error_mode="stderr")
class WorkerControl(MagicTemplate):
    # A widget that has a napari worker object and appears as buttons in the activity dock 
    # while running.
    
    info = vfield(str, enabled=False, record=False)
    
    def __post_init__(self):
        self.paused = False
        self._worker: Worker = None
        self._last_info = ""
    
    @property
    def running(self) -> bool:
        """Return true if worker is running."""
        return self._worker is not None and self._worker._running
    
    def _set_worker(self, worker: Worker):
        """Set worker if no worker is running."""
        if self.running:
            e = Exception(
                "An worker is already running! Please wait until it finishs, or click "
                "the 'Interrupt' button to abort it."
            )
            QtErrorMessageBox.raise_(e, parent=self.native)
            return
        self._worker = worker
        viewer = self.parent_viewer
        viewer.window._status_bar._toggle_activity_dock(True)
        dialog = viewer.window._qt_window._activity_dialog
        
        @worker.finished.connect
        def _on_finish(*args):
            self.info = ""
            viewer.window._status_bar._toggle_activity_dock(False)
            dialog.layout().removeWidget(self.native)
            self.native.setParent(None)

        dialog.layout().addWidget(self.native)
        worker.start()
        return None

    def Pause(self):
        """Pause/Resume thread."""        
        if not isinstance(self._worker, GeneratorWorker):
            return
        if self.paused:
            self._worker.resume()
            self["Pause"].text = "Pause"
            self.info = self._last_info
        else:
            self._worker.pause()
            self["Pause"].text = "Pausing"
            self._last_info = self.info
            @self._worker.paused.connect
            def _on_pause():
                self["Pause"].text = "Resume"
                
        self.paused = not self.paused
        return None
        
    def Interrupt(self):
        """Interrupt thread."""
        self.paused = False
        self["Pause"].text = "Pause"
        self.info = ""
        self._worker.quit()
        return None
