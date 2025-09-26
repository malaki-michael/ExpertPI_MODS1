import ctypes
from threading import Thread
from time import perf_counter


class ProcessingThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = 0
        self.running = False
        self.paused = False

    def start(self):
        self.start_time = perf_counter()
        self.running = True
        super().start()

    def pause(self, paused):
        self.paused = paused

    def stop(self):
        self.running = False

    def force_stop(self, on_stop=None):
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self.ident), ctypes.py_object(Exception))
        if on_stop is not None:
            on_stop()
