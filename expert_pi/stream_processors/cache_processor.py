import threading
import time
import traceback


class Receiver:
    def __init__(self, cache_client, consumers):
        self.thread = None
        self.running = False
        self.output_item = None  # use a single item queue

        self.cache_client = cache_client

        self.batch_size = 4096
        self.scan_id = None
        self.total_pixels = 0
        self.consumers = consumers

    def start(self, scan_id, total_pixels):
        self.scan_id = scan_id
        self.total_pixels = total_pixels

        if self.thread is not None and self.running:
            self.stop()
            if self.thread.is_alive():
                self.thread.join()

        self.thread = threading.Thread(target=self._run)
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.cache_client.disconnect()

    def _run(self):
        remaining = self.total_pixels
        while remaining > 0 and self.running:
            print("recv", remaining)
            start = time.perf_counter()
            to_read = min(remaining, self.batch_size)
            header, data = self.cache_client.get_item(self.scan_id, to_read, raw=True)
            end = time.perf_counter()
            for consumer in self.consumers:
                try:
                    # the concumers can be slower than receiver thread, in that case the cache readout is slowed down
                    consumer(header, data)

                except Exception as _:
                    traceback.print_exc()
            remaining -= to_read
            consumers = time.perf_counter()
            print(f"recv: {(end - start):8.6f} consumers: {(consumers - end):8.6f} ")
