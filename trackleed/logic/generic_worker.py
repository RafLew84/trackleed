from __future__ import annotations

# stal/logic/generic_worker.py
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import traceback

class GenericWorker(QObject):
    """A generic worker to run any function in a background thread."""
    finished = pyqtSignal(object) # Can emit any result type
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.thread: QThread | None = QThread.currentThread()

    def run(self):
        try:
            # Pass the progress signal emitter as a callback to the function
            self.kwargs['progress_callback'] = self.progress.emit
            result = self.function(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")
