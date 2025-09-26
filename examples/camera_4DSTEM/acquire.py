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

folder = "c:/temp/"

# BF image:
N = 1024
grpc_client.projection.set_is_off_axis_stem_enabled(True)
scan_id = scan_helper.start_rectangle_scan(pixel_time=1e-6, total_size=N, frames=1, detectors=[DetectorType.BF])
header, data = cache_client.get_item(scan_id, N**2)
with open(f"{folder}4D_stem_BF.pdat", "wb") as f:
    pickle.dump((header, data), f)

# 4D stem data:

N = 32
scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/4500, 8), total_size=N, frames=1, detectors=[DetectorType.Camera])
header, data = cache_client.get_item(scan_id, N**2)
with open(f"{folder}4D_stem_4500fps.pdat", "wb") as f:
    pickle.dump((header, data), f)

N = 32
grpc_client.projection.set_is_off_axis_stem_enabled(False)
sleep(0.2)  # stabilization
scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/2240, 8), total_size=N, frames=1, detectors=[DetectorType.Camera])
header, data = cache_client.get_item(scan_id, N**2)
with open(f"{folder}4D_stem_2250fps.pdat", "wb") as f:
    pickle.dump((header, data), f)


grpc_client.illumination.set_illumination_values(current=0.5, angle=13)


%%threaded
# long acquisition with precession:
N = 256
folder2="precession/"
try:
    os.mkdir(f"{folder}{folder2}")
except:
    pass
grpc_client.projection.set_is_off_axis_stem_enabled(False)
sleep(0.2)  # stabilization
scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/4500, 8), total_size=N, frames=1, detectors=[DetectorType.Camera],is_precession_enabled=True)
for i in range(N):
    print("getting data",i,"/",N)
    header, data = cache_client.get_item(scan_id, N) # in batches
    for j in range(N):
        file=f"{folder}{folder2}4D_stem_precession_{i*N+j:06}.tiff"
        params = [259, 1]
        cv2.imwrite(file, data["cameraData"][j], params)
