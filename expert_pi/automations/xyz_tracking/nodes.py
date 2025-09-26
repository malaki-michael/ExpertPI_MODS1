import threading
from time import sleep
import traceback
import time
from . import objects


class NodesManager:
    def __init__(self, nodes):
        self.nodes = nodes

        self.running = False

    def start(self):
        self.running = True
        for n in self.nodes:
            n.finished_event.clear()
            n.start()

    def stop(self, wait=True):
        for n in self.nodes:
            n.stop()
        if wait and self.running:
            for n in self.nodes:
                n.finished_event.wait()
        self.running = False

    def print_status(self):
        for i, n in enumerate(self.nodes):
            print(i, n._running)

    def get_node_by_name(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None


class Node:
    def __init__(self, job, name="", event_timeout=None):
        self.name = name
        self.input = None
        self.job = job
        self.output_nodes = []
        self._event = threading.Event()
        self.finished_event = threading.Event()
        self.finished_event.set()
        self._running = False
        self._thread = None
        self.statistic = []
        self.debug_trackers = []
        self.event_timeout = event_timeout  # if None it will wait for event triggered

    def start(self):
        if self._running or (self._thread is not None and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run)
        self._running = True
        # self.statistic = []
        self.finished_event.clear()
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._running = False
            self._event.set()

    def set_input(self, input):
        self.input = input
        self._event.set()

    def _run(self):        
        while True:
            start = time.perf_counter()
            timeout = self._event.wait(timeout=self.event_timeout)
            self._event.clear()
            if not self._running:
                break
            input = self.input  # need to shift it to separate variable
            if input is not None:
                try:
                    job_start = time.perf_counter()
                    diagnostic = objects.Diagnostic(job_start)

                    for debug_tracker in self.debug_trackers:
                        debug_tracker.set_input_signal.emit(input)

                    output = self.job(input)
                    diagnostic.job_time = time.perf_counter() - job_start
                    if output is not None:
                        output.nodes_diagnostics.append((self.name, diagnostic))

                    for n in self.output_nodes:                        
                        n.set_input(output)

                    diagnostic.loop_time = time.perf_counter() - start

                    for debug_tracker in self.debug_trackers:
                        debug_tracker.set_output_signal.emit(diagnostic, output)

                except:
                    traceback.print_exc()
                    # for debug_tracker in self.debug_trackers:
                    #     debug_tracker.set_error_signal.emit()
                    self._running = False
                    break
        self.finished_event.set()
