#find a proper place on the sample and copy script below to the ipython console:
#%%threaded magic comand => make script threaded to do not interfere with the ui

%%threaded
from expert_pi import grpc_client, scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules.scanning import DetectorType as DT
import pickle

folder = "c:/temp/"
file_name = "edx_spectra.pdat"


N = 1024
pixel_time = 1e-6
frames = 60
detectors = [DT.BF, DT.EDX0, DT.EDX1]
print("acquisition time:", N**2*pixel_time*frames, "s")

grpc_client.xray.set_xray_filter_type(grpc_client.xray.XrayFilterType.EnergyResolution)
scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=N, frames=frames, detectors=detectors)
header, data = cache_client.get_item(scan_id, N*N*frames)
with open(folder + file_name, "wb") as f:
    pickle.dump((header, data), f)
