from PySide6 import QtCore


class IdleRunner(QtCore.QObject):
    run = QtCore.Signal(object, tuple, float)

    def __init__(self):
        super().__init__()
        self.run.connect(self.on_run)

    def on_run(self, func, args, delay):
        QtCore.QTimer.singleShot(delay * 1000, lambda: func(*args))


_idle_runner = IdleRunner()


# use this to schedule calls from separate thread to main Qt thread:
def run_on_idle(func, *args, delay=0):
    _idle_runner.run.emit(func, args, delay)
