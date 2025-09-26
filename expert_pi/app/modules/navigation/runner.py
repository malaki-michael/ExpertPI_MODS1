import copy
import threading

import numpy as np

from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.app.modules.navigation import acquisition, scheduler
from expert_pi.config import NavigationConfig
from expert_pi.stream_clients import CacheClient


class NavigationRunner:
    def __init__(
        self,
        cache,
        update_tiles_function,
        update_info_function,
        config: NavigationConfig,
        cache_client: CacheClient,
        update_stage_info_function=None,
    ):
        self.cache = cache
        self.update_tiles_function = update_tiles_function
        self.update_info_function = update_info_function
        self.config = config
        self.cache_client = cache_client
        self.update_stage_info_function = update_stage_info_function

        self.running = False
        self._acquisition_thread = None
        self._reloading_thread = None
        self._event = threading.Event()
        self._loading_event = threading.Event()

        self.actual_tile_parameters = None
        self.tile_job_id = 0
        self.actual_tile_job: scheduler.TileJob | None = None
        self.tile_job_lock = threading.Lock()

        self.enable_acquisition = False

        self.final_stage_position = None

    def view_rectangle_changed(self, xy0, xy2, z, ij0, ij2, force=True):
        max_fov = (
            grpc_client.scanning.get_field_width_range(self.config.pixel_time * 1e-6, self.config.tile_n)["end"] * 1e6
        )

        max_fov = min(self.config.max_fov, max_fov / self.config.tile_overlap)  # limit due to LM mode

        if (
            self.actual_tile_parameters is not None
            and self.actual_tile_job is not None
            and self.actual_tile_job.z == z
            and self.actual_tile_job.ij0[0] == ij0[0]
            and self.actual_tile_job.ij0[1] == ij0[1]
            and self.actual_tile_job.ij2[0] == ij2[0]
            and self.actual_tile_job.ij2[1] == ij2[1]
            and not force
        ):
            return
        self.actual_tile_parameters = (xy0, xy2, z, ij0, ij2, max_fov)
        self._loading_event.set()

    def start(self):
        self.running = True

        # if not ignore_off_axis:
        # if grpc_client.stem_detector.get_is_inserted(grpc_client.stem_detector.DetectorType.BF):
        #     if grpc_client.projection.get_is_off_axis_stem_enabled():
        #         grpc_client.projection.set_is_off_axis_stem_enabled(False)
        # else:
        #     if not grpc_client.projection.get_is_off_axis_stem_enabled():
        #         grpc_client.projection.set_is_off_axis_stem_enabled(True)

        if self._acquisition_thread is None or not self._acquisition_thread.is_alive():
            self._acquisition_thread = threading.Thread(target=self._run)
            self._acquisition_thread.start()

        if self._reloading_thread is None or not self._reloading_thread.is_alive():
            self._reloading_thread = threading.Thread(target=self._run_reloading)
            self._reloading_thread.start()

    def stop(self):
        self.running = False
        self._event.set()
        self._loading_event.set()

    def _run_reloading(self):
        while self.running:
            self._loading_event.wait()
            self._loading_event.clear()

            if not self.running:
                break

            actual_tile_parameters = self.actual_tile_parameters

            if actual_tile_parameters is None:
                continue

            xy0, xy2, z, ij0, ij2, max_fov = actual_tile_parameters
            with self.tile_job_lock:
                self.actual_tile_job = scheduler.TileJob(self.cache, xy0, xy2, z, ij0, ij2, max_fov, self.config)
                self.tile_job_id += 1

            if self.enable_acquisition:
                self._event.set()

            # ---------load cached items-----------------
            tile_ids = self.actual_tile_job.get_all_tiles()
            selection_range = (self.actual_tile_job.z, self.actual_tile_job.ij0, self.actual_tile_job.ij2)

            tile_images = []
            for tid in tile_ids:
                tile_images.append(self.cache.get_tile(*tid))

            self.update_tiles_function(tile_ids, tile_images, selection_range)

    def _run(self):
        while self.running:
            self._event.wait()
            self._event.clear()

            if not self.running:
                break

            with self.tile_job_lock:
                if self.actual_tile_job is None:
                    continue
                actual_tile_job_id = self.tile_job_id * 1

            while self.enable_acquisition and self.running:
                last_electronic, last_stage, _rotation, transform2x2 = scan_helper.get_scanning_shift_and_transform()
                actual_xy = last_stage + transform2x2 @ last_electronic  # sample coordinates

                with self.tile_job_lock:
                    if actual_tile_job_id < self.tile_job_id:
                        break  # break moving to next scheduled job
                    try:
                        new_tiles = self.actual_tile_job.get_next(actual_xy)
                        selection_range = (self.actual_tile_job.z, self.actual_tile_job.ij0, self.actual_tile_job.ij2)
                    except:
                        import traceback

                        traceback.print_exc()

                if new_tiles is None:
                    break

                stage_pos, z_output, tile_ids, to_acquire = new_tiles[0], new_tiles[1], new_tiles[2], new_tiles[3]
                self.move_to_stage_position(np.array(stage_pos))
                if not self.running:
                    break

                done = 0
                for tid in tile_ids:
                    self.update_info_function(to_acquire - done, stage_pos[0], stage_pos[1])
                    done += 1

                    z, i, j = tid
                    tile_rect = self.cache.tile_rect(z, i, j)

                    # including overlaps
                    tile_images = acquisition.acquire_tile(
                        tile_rect, z, i, j, stage_pos, self.config, self.cache_client
                    )
                    tile_images2 = self.cache.write_tile_with_overlaps(*tid, tile_images)

                    self.cache.fill_upper_layers(*tid, copy.deepcopy(tile_images2))  # todo fill up also overlaps

                    if not self.running:
                        break

                    if z_output < tid[0]:
                        z_add = tid[0] - z_output
                        z = z_output
                        i = tid[1] // 2**z_add
                        j = tid[2] // 2**z_add

                        tid = (z, i, j)
                        tile_image = self.cache.get_tile(z, i, j)
                    else:
                        tile_image = tile_images2["BF"]  # TODO channels

                    self.update_tiles_function([tid], [tile_image], selection_range)

        if self.final_stage_position is not None:
            self.move_to_stage_position(self.final_stage_position)
        self.running = False

    def move_to_stage_position(self, stage_position):
        print("moving stage", stage_position)

        last_electronic, last_stage, _rotation, transform2x2 = scan_helper.get_scanning_shift_and_transform()
        actual_xy = last_stage + transform2x2 @ last_electronic  # sample coordinates
        shift_xy = stage_position - actual_xy

        _new_stage, _new_electronic, future, use_stage = scan_helper.set_combined_shift(
            shift_xy, last_electronic, last_stage, transform2x2
        )
        if use_stage:
            if self.update_stage_info_function is not None:
                while future.running():
                    xy_s = grpc_client.stage.read_x_y_z_a_b()
                    xy_stage = np.array([xy_s["x"], xy_s["y"]]) * 1e6  # to um
                    self.update_stage_info_function(xy_stage)
            else:
                future.result()  # wait for move to finish
