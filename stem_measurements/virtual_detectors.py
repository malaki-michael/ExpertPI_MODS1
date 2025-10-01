import numpy as np

from stem_measurements.core import virtual_detector_core, VirtualDetectorCore, VirtualDetectorCompressedCore
import stem_measurements.core

__all__ = ["VirtualDetector", "VirtualDetCompressed", "virtual_detector", "set_detector_mask", "reset_detector_mask"]


class VirtualDetector:
    """class for virtual detector calculation of multiple images."""

    def __init__(self, thread_count=1) -> None:
        """Initialize the virtual detector.

        Args:
            thread_count (int): The number of threads to use. 0 means all available threads.
        """
        self._core = VirtualDetectorCore(thread_count=thread_count)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Virtual detector calculation of multiple images.

        Args:
            images (np.ndarray): The images to calculate vitrual detectors.

        Returns:
            np.ndarray: The virtual detectors. The shape is (number of masks, number of images).
        """
        return self._core(images)


class VirtualDetCompressed:
    """class for virtual detector calculation of multiple images."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype, thread_count=1) -> None:
        """Initialize the virtual detector.

        Args:
            shape (tuple[int, int]): The shape of the images.
            dtype (np.dtype): The data type of the images.
            thread_count (int): The number of threads to use. 0 means all available threads.
        """
        self._core = VirtualDetectorCompressedCore(shape, dtype, thread_count=thread_count)

    def __call__(self, data: np.ndarray, lengths: list[int]) -> np.ndarray:
        """Virtual detector calculation of multiple images.

        Args:
            data (np.ndarray): The compressed images to calculate vitrual detectors.
            lengths (list[int]): The lengths of the compressed images.

        Returns:
            np.ndarray: The virtual detectors. The shape is (number of masks, number of images).
        """
        return self._core(data, lengths)


def virtual_detector(images: np.ndarray, thread_count: int = 1) -> np.ndarray:
    """Sum the images in the virtual detector.

    Args:
        images (np.ndarray): The images to sum.
        thread_count (int): The number of threads to use. 0 means all available threads.

    Returns:
        np.ndarray | list[np.ndarray]: The sum of the images in the virtual detector.
    """
    return virtual_detector_core(images, thread_count=thread_count)


def set_detector_mask(masks: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    """Set the detector masks for the virtual detector and center of mass calculation.

    Single mask is numpy array of the same shape as the detector. Non zero values or True values are considered as
    active pixels.

    Args:
        masks (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): The detector masks or multiple detector masks.
    """
    if isinstance(masks, np.ndarray):
        masks = (masks,)

    masks_final = np.stack(masks).astype(np.uint8)
    stem_measurements.core.set_mask_det(masks_final)


def reset_detector_mask() -> None:
    """Reset the detector mask to the full detector.

    This is the default state of the detector mask. All pixels are considered active pixels.
    """
    stem_measurements.core.reset_mask_det()
