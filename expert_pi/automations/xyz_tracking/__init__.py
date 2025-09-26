import numpy as np
from . import nodes

from .. import settings
from . import acquisition
from . import registration
from . import regulation
from . import stage_shifting
from . import alpha_scheduler

from . import objects

# from ...stream_clients import cache_client
from ...stream_processors import functions
from ...gui.elements.idle_runner import run_on_idle
from ... import grpc_client
from expert_pi.app import scan_helper

manager = None


def setup_loop():
    global manager
    if manager is not None:
        manager.stop()

    acquisition_node = nodes.Node(acquisition.acquire_next, "acquisition")  # dx,dy -> img, fov, scale

    registration_node = nodes.Node(registration.get_shift, "registration")  # image, fov, scale -> x, cx, y, cy
    acquisition_node.output_nodes.append(registration_node)

    image_view_node = nodes.Node(lambda x: None, "image_viewing")  # img, fov, scale ->None # insert by the code
    registration_node.output_nodes.append(image_view_node)

    regulation_node = nodes.Node(regulation.PID_stage_offset, "regulation")  # x, cx, y, cy -> dx, dy, dz
    registration_node.output_nodes.append(regulation_node)

    stage_node = nodes.Node(stage_shifting.compensate_stage, "stage_shift", event_timeout=settings.stage_shifting_timeout)  # dx, dy, dz -> error
    regulation_node.output_nodes.append(stage_node)

    # saving_node = nodes.Node(stage_shifting.add_datapoint, "saving_node")
    # stage_node.output_nodes.append(saving_node)

    stage_node.output_nodes.append(acquisition_node)

    node_list = [
        acquisition_node,
        registration_node,
        regulation_node,
        stage_node,
        # saving_node,
        image_view_node
    ]

    manager = nodes.NodesManager(node_list)


def acquire_reference(fov, image_view=None, fovs=None, fovs_total=None,
                      total_pixels=None, rectangles=None, offsets=None):
    if fov <= 0:
        raise Exception("wrong fov", fov)
    if fovs is None:
        fov_max_ref = grpc_client.scanning.get_field_width_range(settings.pixel_time*1e-9, settings.reference_N)["end"]*1e6
        fov_max_track = grpc_client.scanning.get_field_width_range(settings.pixel_time*1e-9, settings.tracking_N)["end"]*1e6
        if fov > fov_max_track or fov > fov_max_ref:
            raise Exception(f"fov {fov} must be less then {fov_max_track} and {fov_max_track}")

        fovs = []
        fov_add = fov_max_ref
        while fov_add > 1.5*fov:
            fovs.append(fov_add)
            fov_add /= 2
        fovs.append(fov)

    if fovs_total is None:
        fovs_total = np.array(fovs)
        total_pixels = np.array([settings.reference_N]*len(fovs))
        rectangles = np.array([[0, 0, settings.reference_N, settings.reference_N]]*len(fovs))
        offsets = np.array([[0, 0]]*len(fovs))

    settings.allowed_fovs = np.array(fovs)
    images = []
    absolute_positions = []

    shift_electronic, stage_xy, rotation, transform2x2 = scan_helper.get_scanning_shift_and_transform()
    stage_z = grpc_client.stage.get_z()*1e6
    stage_ab = np.array([grpc_client.stage.get_alpha(), grpc_client.stage.get_beta()])

    for i, fov in enumerate(fovs):
        if rectangles is not None:
            image = acquisition.acquire_image(fovs_total[i], total_pixels[i], settings.pixel_time, rectangle=rectangles[i])
        else:
            image = acquisition.acquire_image(fov, settings.reference_N, settings.pixel_time)
        if image_view is not None:
            image_view.raw_image = image
            image_view.shape = image.shape
            image_8b = functions.calc_cast_numba(image_view.raw_image.ravel(), image_view.alpha, image_view.beta)
            image_8b = image_8b.reshape(image_view.shape)
            image_view.image_item.set_image(image_8b)
            image_view.image_item.fov = fov
            image_view.image_item.shift = [0, 0]
            # needs to be run from main ui thread:
            image_view.redraw()
        images.append(image)

    reference = objects.Reference(fovs, images)
    reference.shift_electronic = shift_electronic
    reference.stage_xy = stage_xy
    reference.rotation = rotation
    reference.scanning_to_sample_transform = transform2x2
    reference.stage_z = stage_z
    reference.stage_ab = stage_ab

    reference.fovs_total = fovs_total
    reference.total_pixels = total_pixels
    reference.rectangles = rectangles
    reference.offsets = offsets

    registration.set_reference(reference)

    regulation.actual_errors_integral = None
    regulation.actual_errors = None
    stage_shifting.manual_xyz_correction = np.zeros(3)

    return reference


def start_tracking(fov):
    global manager

    registration.initialize()
    registration.series += 1

    settings.requested_fov = fov
    regulation.actual_errors_integral = None
    regulation.actual_errors = None

    cache_client.connect()
    for node in manager.nodes:
        node.input = None
    manager.start()

    input = objects.NodeImage(0, registration.series, None, None, None)
    input.next_fov = fov

    manager.nodes[0].set_input(input)  # TODO named it


def stop_tracking(image_view=None):
    manager.stop()
    cache_client.disconnect()


setup_loop()
