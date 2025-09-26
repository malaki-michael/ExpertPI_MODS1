import numpy as np
import cv2
from .. import settings
from enum import Enum

from expert_pi.measurements import shift_measurements


class MotionType(Enum):
    moving = 0
    stabilizing = 1
    waiting = 3
    acquiring = 4
    acquired = 5


class Diagnostic:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.job_time = 0
        self.loop_time = 0


class NodeImage:
    def __init__(self, id, serie_id, image, fov, pixel_time):
        self.id = id
        self.serie_id = serie_id
        self.image = image
        self.fov = fov
        self.pixel_time = pixel_time

        self.rectangle_selection = False
        self.fov_total = None
        self.total_pixels = None
        self.rectangle = None
        self.center = None
        self.reference_N = None

        # for tomography acquisition
        self.motion_type = MotionType.moving
        self.motion_type_timer = None
        self.acquire_at_this_step = False

        # added by acquisition
        self.stage_xy = None  # um
        self.stage_z = None  # um
        self.stage_ab = None  # rad
        self.transform2x2 = None
        self.shift_electronic = None  # um
        self.rotation = None
        self.acquisition_time_counters = None
        self.scale = 1  # scale with respect to the reference

        self.tilt_angle = None  # mrad

        # added by registration
        self.reference_object: Reference = None
        self.offset_xy = None  # um
        self.offset_z = None  # um
        self.offset_z_xy_shift = None
        self.confidence_xy = None
        self.confidence_z = None

        self.correlation_xy = None
        self.std_x = 0
        self.std_y = 0

        # added by regulation
        self.target_z_offset = None
        self.target_xy_offset = None

        self.errors = None  # numpy xyz um
        self.errors_integral = None
        self.errors_difference = None

        self.previous_errors = None  # numpy xyz um
        self.previous_errors_integral = None
        self.previous_errors_difference = None

        self.regulated_xyz_shift = None
        self.total_error = None  # um

        # added by stage_shifting
        self.next_xy = None  # um
        self.next_z = None  # um
        self.next_fov = None  # um
        self.xyz_correction = None

        self.nodes_diagnostics = []


class Reference:
    def __init__(self, fovs, images):
        self.images = {}
        self.images_normalized = {}

        # need to be filled
        self.shift_electronic = None
        self.stage_xy = None
        self.rotation = None
        self.scanning_to_sample_transform = np.eye(2)
        self.stage_z = None
        self.stage_ab = None

        self.fovs_total = None
        self.total_pixels = None
        self.rectangles = None
        self.offsets = None

        for i, fov in enumerate(fovs):
            self.images[fov] = images[i]
            self.images_normalized[fov] = (images[i]/np.max(images[i])*255).astype(np.uint8)

    def transform_image(self, image, from_transform, to_transform):
        """transform 2x2 matrix, return transformed image with the same center and same dimensions"""
        transform = np.linalg.solve(to_transform.T, from_transform.T).T
        warp_b = np.dot(np.eye(2) - transform, np.array(image.shape).reshape(2, 1)/2)
        warp_mat = np.hstack([transform, warp_b])
        warped_image = cv2.warpAffine(image, warp_mat, image.shape)
        return warped_image

    def get_image(self, fov, scanning_to_sample_transform=None):
        image = self.images[fov]
        if scanning_to_sample_transform is not None:
            return self.transform_image(image, self.scanning_to_sample_transform, scanning_to_sample_transform)
        return image

    def get_offset(self, fov, image, scanning_to_sample_transform=None, coordinates_type="reference", **kwargs):
        """ coordinates_type: reference or image"""
        reference_image = self.images_normalized[fov]

        if scanning_to_sample_transform is not None:
            if coordinates_type == "reference":
                image = self.transform_image(image, scanning_to_sample_transform, self.scanning_to_sample_transform)
            elif coordinates_type == "image":
                reference_image = self.transform_image(reference_image, self.scanning_to_sample_transform, scanning_to_sample_transform)

        shift, corr_coeff = shift_measurements.get_offset_of_pictures(reference_image, image, fov, method=shift_measurements.Method[settings.model], get_corr_coeff=True, **kwargs)
        return shift, corr_coeff

    def get_correlation_coeffs(self, fov, image, shifts, scanning_to_sample_transform=None, coordinates_type="reference", **kwargs):
        """ coordinates_type: reference or image"""
        reference_image = self.images_normalized[fov]

        if scanning_to_sample_transform is not None:
            if coordinates_type == "reference":
                image = self.transform_image(image, scanning_to_sample_transform, self.scanning_to_sample_transform)
            elif coordinates_type == "image":
                reference_image = self.transform_image(reference_image, self.scanning_to_sample_transform, scanning_to_sample_transform)

        ref_for_coeff = shift_measurements.kernels.image_function(reference_image, "gaussian", crop=True)
        img_for_coeff = shift_measurements.kernels.image_function(image, "gaussian", crop=True)

        results = []
        for shift in shifts:
            corr_coeff = shift_measurements.get_correlation_coefficient(ref_for_coeff, img_for_coeff, [-shift[1]/fov*image.shape[0], shift[0]/fov*image.shape[0]], **kwargs)
            results.append(corr_coeff)
        return results
