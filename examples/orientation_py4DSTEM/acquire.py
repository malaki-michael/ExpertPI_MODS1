%%threaded

from expert_pi.stream_clients import cache_client
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules.scanning import DetectorType as DetectorType
import pickle
import numpy as np
from time import sleep
import cv2
import os
import h5py

folder = "d:/pystem_cache/"

# BF image:
N = 1024
grpc_client.projection.set_is_off_axis_stem_enabled(True)
scan_id = scan_helper.start_rectangle_scan(pixel_time=1e-6, total_size=N, frames=1, detectors=[DetectorType.BF])
header, data = cache_client.get_item(scan_id, N**2)
with open(f"{folder}4D_stem_BF.pdat", "wb") as f:
    pickle.dump((header, data), f)

# # 4D stem data:
#
# N = 32
# scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/4500, 8), total_size=N, frames=1, detectors=[DT.Camera])
# header, data = cache_client.get_item(scan_id, N**2)
# with open(f"{folder}4D_stem_4500fps.pdat", "wb") as f:
#     pickle.dump((header, data), f)
#
# N = 32
# grpc_client.projection.set_is_off_axis_stem_enabled(False)
# sleep(0.2)  # stabilization
# scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/2240, 8), total_size=N, frames=1, detectors=[DT.Camera])
# header, data = cache_client.get_item(scan_id, N**2)
# with open(f"{folder}4D_stem_2250fps.pdat", "wb") as f:
#     pickle.dump((header, data), f)
#




%%threaded
# long acquisition with precession:
N = 256

grpc_client.projection.set_is_off_axis_stem_enabled(False)
sleep(0.2)  # stabilization
scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/2000, 8), total_size=N, frames=1, detectors=[DetectorType.Camera],is_precession_enabled=True)




batch_size=N
pixels_done = 0
total_pixels = N**2

depth_8bit=False

with h5py.File(f"{folder}4D_stem_precession.h5", 'w') as file:
    while pixels_done < N**2:
        pixels_to_query = batch_size if (total_pixels - pixels_done) > batch_size else total_pixels - pixels_done
        header, data = cache_client.get_item(scan_id, pixel_count=pixels_to_query, raw=True)
        images = cache_client.parse_camera_data(header, data)

        if pixels_done == 0:
            data_type = np.uint8 if images.dtype == np.uint8 or depth_8bit else images.dtype
            data_type = np.uint16 if data_type == np.uint32 else data_type

            file.create_dataset('data', shape=(total_pixels, 512*512), dtype=data_type, chunks=(min(total_pixels, batch_size), 16), compression='lzf')
            file.create_dataset('size', (2,), dtype=np.uint16)
            file['size'][:] = [N,N]

        if images.dtype == np.uint32:
            images = np.clip(images, 0, 65535).astype(np.uint16)

        if depth_8bit and images.dtype == np.uint16:
            images = cv2.convertScaleAbs(images)

        if images.shape[1] == 514:
            images = images[:, 1:513, 1:513].reshape(header['pixelCount'], 512*512)
        else:
            images = images.reshape(header['pixelCount'], 512*512)
        file['data'][pixels_done:pixels_done + header['pixelCount']] = images[:, :]

        pixels_done += header['pixelCount']
        print(f'{pixels_done} / {total_pixels} ({pixels_done/total_pixels*100:.1f} %)')