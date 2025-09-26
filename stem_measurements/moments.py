import numpy as np

import stem_measurements.core
from stem_measurements.core import center_of_mass_core


def set_detector_mask(masks: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    """Set the detector masks for the virtual detector and center of mass calculation.

    Single mask is numpy array of the same shape as the detector. Non zero values or True values are considered as
    active pixels.

    Args:
        masks (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): The detector masks or multiple detector masks.
    """
    if isinstance(masks, np.ndarray):
        masks = (masks,)

    starts_all, ends_all = [], []
    ys, x_starts, x_ends = [], [], []
    for mask in masks:
        mask_flatt = mask.copy().astype(bool).flatten()
        maske = np.concatenate([[False], mask_flatt, [False]]).astype(np.int8)
        mm = maske[1:] - maske[:-1]
        starts = np.nonzero(mm == 1)[0].astype(np.uint64)
        ends = np.nonzero(mm == -1)[0].astype(np.uint64)

        starts_all.append(starts)
        ends_all.append(ends)

        mask_bool = mask.copy().astype(bool)
        c = np.zeros((mask_bool.shape[0], 1), dtype=bool)
        maske = np.concatenate([c, mask_bool, c], axis=1).astype(np.int8)
        mm = maske[:, 1:] - maske[:, :-1]
        starts = np.transpose(np.nonzero(mm == 1)).astype(np.uint64)
        ends = np.transpose(np.nonzero(mm == -1)).astype(np.uint64)

        ys.append(starts[:, 0].flatten())
        x_starts.append(starts[:, 1].flatten())
        x_ends.append(ends[:, 1].flatten())

    stem_measurements.core.set_mask_det(starts_all, ends_all, ys, x_starts, x_ends)


def reset_detector_mask() -> None:
    """Reset the detector mask to the full detector.

    This is the default state of the detector mask. All pixels are considered active pixels.
    """
    stem_measurements.core.reset_mask_det()


def center_of_mass(images: np.ndarray) -> np.ndarray | list[np.ndarray]:
    """Calculate the center of mass of the images.

    Args:
        images (np.ndarray): The images to calculate the center of mass.

    Returns:
        np.ndarray | list[np.ndarray]: The center of mass of the images.
    """
    result = center_of_mass_core(images)
    if len(result) == 1:
        result = result[0]
    return result
