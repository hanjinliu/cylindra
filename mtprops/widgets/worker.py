"""Widgets and functions that deal with napari's workers."""

from __future__ import annotations
import sys
from typing import Union, Callable, TYPE_CHECKING
if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec
import warnings
from functools import wraps

from napari._qt.qthreading import GeneratorWorker, FunctionWorker

from magicclass import magicclass, vfield, MagicTemplate
from magicclass.gui._message_box import QtErrorMessageBox

if TYPE_CHECKING:
    from .main import MTPropsWidget

Worker = Union[FunctionWorker, GeneratorWorker]


def run_worker_function(worker: Worker):
    """Emulate worker execution."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("always")
            warnings.showwarning = lambda *w: worker.warned.emit(w)
            result = worker.work()
        if isinstance(result, Exception):
            raise result
        worker.returned.emit(result)
    except Exception as exc:
        worker.errored.emit(exc)
    worker._running = False
    worker.finished.emit()
    worker._finished.emit(worker)

_P = ParamSpec("_P")

def dispatch_worker(f: Callable[_P, Worker]) -> Callable[_P, None]:
    """
    Open a progress bar and start worker in a parallel thread if function is called from GUI.
    Otherwise (if function is called from script), the worker will be executed as if the 
    function is directly called.
    """
    @wraps(f)
    def wrapper(self: "MTPropsWidget", *args, **kwargs):
        worker = f(self, *args, **kwargs)
        if self[f.__name__].running:
            self._connect_worker(worker)
            worker.start()
        else:
            run_worker_function(worker)
        return None
    return wrapper


@magicclass(layout="horizontal", labels=False, error_mode="stderr")
class WorkerControl(MagicTemplate):
    # A widget that has a napari worker object and appears as buttons in the activity dock 
    # while running.
    
    info = vfield(str, enabled=False, record=False)
    
    def __post_init__(self):
        self.paused = False
        self.worker: Worker = None
        self._last_info = ""
    
    def _set_worker(self, worker: Worker):
        self.worker = worker
        @worker.errored.connect
        def _(e=None):
            # In some environments, errors raised in workers are completely hidden.
            # We have to re-raise it here.
            QtErrorMessageBox.raise_(e, parent=self.native)
        
    def Pause(self):
        """Pause/Resume thread."""        
        if not isinstance(self.worker, GeneratorWorker):
            return
        if self.paused:
            self.worker.resume()
            self["Pause"].text = "Pause"
            self.info = self._last_info
        else:
            self.worker.pause()
            self["Pause"].text = "Resume"
            self._last_info = self.info
            self.info = "Pausing"
        self.paused = not self.paused
        
    def Interrupt(self):
        """Interrupt thread."""
        self.worker.quit()
