from functools import lru_cache

import cv2
import numpy as np
from sklearn import linear_model

# import config
from expert_pi.automations import settings
from expert_pi.automations.xyz_tracking import objects
from expert_pi.measurements import shift_measurements

reference_image = None

folder = "C:/temp/registration_images2/"
counter = 0
series = 0
save = False

global t

offset_history_buffer = []

ransac = linear_model.RANSACRegressor()


def filter_signal(xyz):
    global offset_history_buffer
    offset_history_buffer.append(xyz)
    # print(xyz)
    while len(offset_history_buffer) > settings.filtering_samples:
        del offset_history_buffer[0]

    return np.median(offset_history_buffer, axis=0)

    # if len(offset_history_buffer) < 3:
    #     return xyz
    #
    # x = np.linspace(0, 1, num=min(len(offset_history_buffer), settings.filtering_samples))
    # #print(x,offset_history_buffer)
    #
    # fit = ransac.fit(x.reshape(-1, 1), np.array(offset_history_buffer))
    # #print(fit.predict(np.array([[1]])))
    # return fit.predict(np.array([[1]]))[0]


def save_pair(ref, image, left_mask, right_mask, fov, pixel_time):
    global counter, t
    x = cv2.hconcat([ref, image])
    y = cv2.hconcat([(left_mask * 255).astype("uint8"), (right_mask * 255).astype("uint8")])
    t = x, y
    name = f"{series:02}_{counter:04}_fov_{fov}_dt_{pixel_time}"
    cv2.imwrite(folder + name + ".tif", x)
    cv2.imwrite(folder + name + "_mask.tif", y)
    counter += 1


def set_reference(reference):
    global reference_image
    reference_image = reference


def get_shift(input: objects.NodeImage) -> objects.NodeImage:
    """Image can be scaled down with respect to reference images image and reference already normalize to 255."""
    global reference_image

    if (
        shift_measurements.Method[settings.model] & shift_measurements.Method.TEMRegistrationAll
        and settings.tem_registration_batch
    ):
        return get_shift_tem_optimized(input)

    if len(input.image.shape) == 3:
        image_normalized = input.image[0] / np.max(input.image[0])
        shape = (np.array(input.image.shape[1:]) * input.scale).astype("int")
        image_shape = input.image.shape[1:]
    else:
        image_normalized = input.image / np.max(input.image)
        shape = (np.array(input.image.shape) * input.scale).astype("int")
        image_shape = input.image.shape

    image8 = (image_normalized * 255).astype("uint8")  # normalization

    image_scaled = np.zeros(shape, dtype="uint8")
    mask = np.zeros(shape, dtype="bool")
    mask[
        shape[0] // 2 - image_shape[0] // 2 : shape[0] // 2 + image_shape[0] // 2,
        shape[0] // 2 - image_shape[1] // 2 : shape[1] // 2 + image_shape[1] // 2,
    ] = True
    image_scaled[mask] = image8.flat

    left_mask = np.zeros((settings.reference_N, settings.reference_N))  # fixed by Tem registration Tiny model
    right_mask = np.ones((settings.reference_N, settings.reference_N))
    right_mask[mask] = 0  # opposite definition of mask for model

    input.reference_object = reference_image

    ref = reference_image.images_normalized[input.fov * input.scale]
    reference_image_transformed = input.reference_object.transform_image(
        ref, input.reference_object.scanning_to_sample_transform, input.transform2x2
    )

    if save:
        save_pair(ref, image_scaled, left_mask, right_mask, input.fov, settings.pixel_time)

    image_rectangle = [
        (settings.reference_N - settings.tracking_N) // 2,
        (settings.reference_N - settings.tracking_N) // 2,
        settings.tracking_N,
        settings.tracking_N,
    ]

    shift, corr_coeff = reference_image.get_offset(
        input.fov * input.scale,
        image_scaled,
        scanning_to_sample_transform=input.transform2x2,
        coordinates_type="image",
        left_masks=left_mask,
        right_masks=right_mask,
        inner_rectangle_1=image_rectangle,
    )

    if len(input.image.shape) == 3:
        shift_z, corr_coeff_z = shift_measurements.get_offset_of_pictures(
            input.image[0],
            input.image[1],
            input.fov,
            method=shift_measurements.Method[settings.model],
            get_corr_coeff=True,
        )
        total_angle = np.sqrt(np.sum(input.tilt_angle**2))
        directional_shift = np.sum(shift_z * input.tilt_angle) / total_angle
        input.offset_z_xy_shift = shift_z
        input.offset_z = directional_shift / np.arctan(total_angle * 1e-3)
        input.confidence_z = corr_coeff_z

    fov = input.fov * input.scale
    # shifts = [
    #     shift + np.array([settings.correlation_shift_factor*fov, 0]),
    #     shift + np.array([-settings.correlation_shift_factor*fov, 0]),
    #     shift + np.array([0, settings.correlation_shift_factor*fov]),
    #     shift + np.array([0, -settings.correlation_shift_factor*fov])
    # ]
    #
    # corr_coeffs = reference_image.get_correlation_coeffs(input.fov*input.scale, image_scaled, shifts,
    #                                                      scanning_to_sample_transform=input.transform2x2, coordinates_type="image",
    #                                                      inner_rectangle_1=image_rectangle)

    if np.isnan(shift).any():
        x, y = 0, 0

    else:
        x = shift[0]
        y = shift[1]

    input.offset_xy = np.array([x, y])

    # input.std_x = corr_coeff - np.mean(corr_coeffs[0:2])
    # input.std_y = corr_coeff - np.mean(corr_coeffs[2:4])

    # input.correlation_xy = corr_coeff

    # input.confidence_xy = corr_coeff*np.array([input.std_x, input.std_y])  # will be -1 if nan  TODO change definition

    input.std_x = 0
    input.std_y = 0

    input.correlation_xy = 1
    input.confidence_xy = np.array([1, 1])

    return input


@lru_cache(maxsize=1)
def prepare_masks(reference_N, tracking_N):
    mask = np.zeros((reference_N, reference_N), dtype="bool")
    ir = [(reference_N - tracking_N) // 2, (reference_N - tracking_N) // 2, tracking_N, tracking_N]
    mask[ir[0] : ir[0] + ir[2], ir[1] : ir[1] + ir[3]] = True

    reference_mask = np.zeros((reference_N, reference_N), dtype=np.uint8)  # will work only for reference_N=512
    image_mask = np.ones((reference_N, reference_N), dtype=np.uint8)
    image_mask[mask] = 0  # opposite definition of mask for model

    return ir, mask, reference_mask, image_mask


temp = None
temp_m = None


def get_shift_tem_optimized(input: objects.NodeImage) -> objects.NodeImage:
    """Image can be scaled down with respect to reference images image and reference already normalize to 255."""
    global reference_image
    if len(input.image.shape) == 3:
        image8 = (input.image[0] / np.max(input.image[0]) * 255).astype(np.uint8)
    else:
        image8 = (input.image / np.max(input.image) * 255).astype(np.uint8)

    ri = input.rectangle

    input.reference_object = reference_image
    ref_i = np.where(reference_image.fovs_total == input.fov_total)[0][0]
    rr = reference_image.rectangles[ref_i]
    ref = reference_image.images_normalized[input.fov]
    ref_full = np.zeros((input.reference_N, input.reference_N), dtype=np.uint8)
    ref_full[: rr[2], : rr[3]] = ref

    ref_mask = np.ones((input.reference_N, input.reference_N), dtype=np.uint8)
    ref_mask[: rr[2], : rr[3]] = 0

    reference_image_transformed = input.reference_object.transform_image(
        ref_full, input.reference_object.scanning_to_sample_transform, input.transform2x2
    )

    image_scaled = np.zeros((input.reference_N, input.reference_N), dtype=np.uint8)
    image_scaled[ri[0] - rr[0] : ri[0] - rr[0] + ri[2], ri[1] - rr[1] : ri[1] - rr[1] + ri[3]] = image8
    image_mask = np.ones((input.reference_N, input.reference_N), dtype=np.uint8)
    image_mask[ri[0] - rr[0] : ri[0] - rr[0] + ri[2], ri[1] - rr[1] : ri[1] - rr[1] + ri[3]] = 0

    image_pairs = [cv2.hconcat([reference_image_transformed, image_scaled])]

    left_masks = [ref_mask]
    right_masks = [image_mask]

    if len(input.image.shape) == 3:
        image_tilted_scaled = np.zeros((input.reference_N, input.reference_N), dtype=np.uint8)
        image_tilted_scaled[ri[0] - rr[0] : ri[0] - rr[0] + ri[2], ri[1] - rr[1] : ri[1] - rr[1] + ri[3]] = (
            input.image[1] / np.max(input.image[1]) * 255
        ).astype(np.uint8)

        image_pairs.append(cv2.hconcat([image_scaled, image_tilted_scaled]))

        left_masks.append(image_mask)
        right_masks.append(image_mask)

    global temp, temp_m
    temp = (image_pairs, right_masks, left_masks)
    try:
        model_output = shift_measurements.registration_model(
            image_pairs,
            host=config.registration_server_host,
            port=config.registration_server_port,
            model_name=shift_measurements.Method[settings.model].name,
            return_only_homography=False,
            left_masks=left_masks,
            right_masks=right_masks,
        )
    except:
        import traceback

        traceback.print_exc()
        model_output = [{"success": False}, {"success": False}]

    shifts = []
    temp_m = model_output
    i = 0
    for m in model_output:
        if not m["success"]:
            input.correlation_xy = -1
            input.confidence_xy = np.array([-1, -1])  # will be -1 if nan  TODO change definition
            shifts = []
            for m in range(len(image_pairs)):
                shifts.append(np.array([0, 0]))
            print("prediction failed")
            break

        if settings.keypoints_matching == "translation":
            xy = np.array(m["translation"]) * input.fov
        else:
            mask = np.array(m["fine_inliers"])
            lp = np.array(m["left_fine_points"])[mask.flat == 1, :]
            rp = np.array(m["right_fine_points"])[mask.flat == 1, :]
            xy_all = lp - rp

            if i == 1 and settings.linear_regression_z:
                X = lp[:, 1] - 0.5
                y = xy_all[:, 0]

                model = linear_model.LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), y)

                y2 = model.predict(X.reshape(-1, 1))

                xy_all[:, 0] -= y2

            if settings.keypoints_matching == "median":
                xy = np.median(xy_all, axis=0) * input.fov
            else:
                xy = np.mean(xy_all, axis=0) * input.fov

        shifts.append(np.array([xy[0], -xy[1]]))  # TODO confidence

        input.correlation_xy = 1
        input.confidence_xy = np.array([1, 1])  # will be -1 if nan  TODO change definition
        i += 1

    if len(input.image.shape) == 3:
        total_angle = np.sqrt(np.sum(input.tilt_angle**2))
        directional_shift = np.sum(shifts[1] * input.tilt_angle) / total_angle
        input.offset_z_xy_shift = shifts[1]
        input.offset_z = directional_shift / np.arctan(total_angle * 1e-3)
        input.confidence_z = 1

    input.offset_xy = shifts[0]

    if settings.offset_filtering:
        xyz = np.array([input.offset_xy[0], input.offset_xy[1], input.offset_z])
        filtered_xyz = filter_signal(xyz)
        input.offset_xy = filtered_xyz[:2]
        input.offset_z = filtered_xyz[2]

    input.std_x = 0
    input.std_y = 0

    return input
