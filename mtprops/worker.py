from __future__ import annotations
from typing import Union, Callable, Any, TYPE_CHECKING
import warnings
from functools import wraps

from napari._qt.qthreading import GeneratorWorker, FunctionWorker

from magicclass import magicclass, vfield, MagicTemplate
from magicclass.gui._message_box import QtErrorMessageBox

if TYPE_CHECKING:
    from .widget import MTPropsWidget

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


def dispatch_worker(f: Callable[[Any], Worker]) -> Callable[[Any], None]:
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
    
### Child widgets ###

@magicclass(layout="horizontal", labels=False, error_mode="stderr")
class WorkerControl(MagicTemplate):
    # A widget that has a napari worker object and appears as buttons in the activity dock 
    # while running.
    
    info = vfield(str, record=False)
    
    def __post_init__(self):
        self.paused = False
        self.worker: Worker = None
        self._last_info = ""
    
    def _set_worker(self, worker):
        self.worker = worker
        @worker.errored.connect
        def _(e):
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
