import ctypes
from threading import Thread

from IPython.core.getipython import get_ipython
from IPython.core.magic import register_cell_magic, register_line_magic


class StoppableThread(Thread):
    def try_stop(self, on_stop=None):
        if self.ident is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self.ident), ctypes.py_object(Exception))
            if on_stop is not None:
                on_stop()


cell_threads = []


def clean_threads():
    keep = []
    global cell_threads
    for thread in cell_threads:
        if thread.is_alive():
            keep.append(thread)

    cell_threads = keep


def start_threaded_function(target):
    clean_threads()
    new_thread = StoppableThread(target=target)
    cell_threads.append(new_thread)
    new_thread.start()
    return new_thread


try:

    @register_cell_magic
    def threaded(line, cell=None):
        ip = get_ipython()
        clean_threads()

        def fn():
            if ip is not None:
                result = ip.run_cell(cell)
                print("done:", result.result)

        print("cell started in thread")
        start_threaded_function(fn)

    @register_line_magic
    def stop_threads(line):
        for thread in cell_threads:
            if thread.is_alive():
                thread.try_stop()

except AttributeError:
    pass
