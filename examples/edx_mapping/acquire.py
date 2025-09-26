# find a proper place on the sample
# use higher current - 5nA to get enough EDX events
# copy script below to the ipython console:
# %%threaded magic command => make script threaded to do not interfere with the ui

%%threaded
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.stream_clients import cache_client
from stem_measurements import shift_measurements
import pickle
import numpy as np
from expert_pi.grpc_client.modules.scanning import DetectorType as DetectorType

folder = "c:/temp/"
file_name = "edx_map"

pixel_time = 6e-6  # s
fov = 0.2  # um
scan_rotation = 0
N = 1024
frames = 50

print("total_time:", pixel_time*N**2*frames/60, "min")

R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
              [-np.sin(scan_rotation), np.cos(scan_rotation)]])

grpc_client.scanning.set_rotation(scan_rotation)
grpc_client.scanning.set_field_width(fov*1e-6)

scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1, detectors=[DetectorType.BF, DetectorType.HAADF, DetectorType.EDX1, DetectorType.EDX0])
header, data = cache_client.get_item(scan_id, N**2)
s0 = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

with open(f"{folder}{file_name}_0.pdat", "wb") as f:
    pickle.dump((header, data, s0), f)

img = data["stemData"]["BF"].reshape(N, N)

for i in range(1, frames):
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=1, detectors=[DetectorType.BF, DetectorType.HAADF, DetectorType.EDX1, DetectorType.EDX0])
    header, data = cache_client.get_item(scan_id, N**2)
    s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    with open(f"{folder}{file_name}_{i}.pdat", "wb") as f:
        pickle.dump((header, data, s), f)

    # apply drift correction between images:
    img2 = data["stemData"]["BF"].reshape(1024, 1024)
    shift = shift_measurements.get_offset_of_pictures(img, img2, fov, method=shift_measurements.Method.PatchesPass2)
    shift = np.dot(R, shift)  # rotate back
    print(i, (s0['x'] - s['x'])*1e9, (s0['y'] - s['y'])*1e9)
    grpc_client.illumination.set_shift({"x": s['x'] - shift[0]*1e-6, "y": s['y'] - shift[1]*1e-6}, grpc_client.illumination.DeflectorType.Scan)
