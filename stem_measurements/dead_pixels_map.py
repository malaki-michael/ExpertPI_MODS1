from typing import TypeAlias

import numpy as np

from stem_measurements import core

BadPixelsInput: TypeAlias = list[tuple[tuple[int, int], list[tuple[int, int]]]]


def set_dead_pixels(dead_pixels: np.ndarray, shape: tuple[int, int] = (512, 512)) -> None:
    """Global dead pixels correction.

    Dead pixels are corrected by averaging the values of the 8 neighbours. It se globaly for whole pagackage.

    Args:
        dead_pixels (np.ndarray): Array of dead pixels coordinates in format np.array([[y1,x1],[y2,x2],...]).
        shape (tuple[int, int], optional): Shape of the image. Defaults to (512, 512).
    """
    dead_pixels_map = generate_correction_map(dead_pixels, shape)
    core.set_bad_pixels_mask(dead_pixels_map, shape)


def reset_dead_pixel():
    """Reset global dead pixels correction."""
    core.reset_bad_pixels_mask()


def generate_correction_map(dead_pixels: np.ndarray, shape: tuple[int, int] = (512, 512)) -> BadPixelsInput:
    """Generate dead pixels correction map list of dead pixels.

    Args:
        dead_pixels (np.ndarray): Array of dead pixels coordinates in format np.array([[y1,x1],[y2,x2],...]).
        shape (tuple[int, int], optional): Shape of the image. Defaults to (512, 512).

    Returns:
        BadPixelsInput: Dictionary of dead pixels and their neighbours.
    """
    # all 8 neighbours, equal weight to skip the recalculation to float
    neighbours = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    dead_pixels_neighbours = []
    while len(dead_pixels) > 0:
        remaining_dead_pixels = []
        for pixel in dead_pixels:
            valid_neighbours = []
            for neighbour in neighbours:
                test_pixel = pixel + neighbour
                if 0 <= test_pixel[0] < shape[0] and 0 <= test_pixel[1] < shape[1]:
                    if list(test_pixel) not in dead_pixels.tolist():
                        valid_neighbours.append(tuple(test_pixel))

            if len(valid_neighbours) > 0:
                dead_pixels_neighbours.append(((pixel[0], pixel[1]), valid_neighbours))
            else:
                remaining_dead_pixels.append(
                    pixel
                )  # store pixels which in this iteration does not have any valid neighbour
        if len(remaining_dead_pixels) == len(dead_pixels):
            raise RuntimeError(
                "map generation failed"
            )  # just in case somebody would attempt to have all the pixels in the frame marked as dead

        dead_pixels = np.array(remaining_dead_pixels)
        # print("remaining:", len(remaining_dead_pixels))
    return dead_pixels_neighbours


def correct_dead_pixels(image, dead_pixels_neighbours) -> np.ndarray:
    """Correct dead pixels in image. This function serves as a reference implementation.

    Args:
        image (np.ndarray): Image to be corrected.
        dead_pixels_neighbours (BadPixelsInput): Dictionary of dead pixels and their neighbours.

    Returns:
        np.ndarray: Corrected image.
    """
    for pixel, neighbours in dead_pixels_neighbours.items():
        value = 0  # python int 64, no problem with overflow
        for n in neighbours:
            value += image[n[0], n[1]]
        if len(neighbours) > 0:  # just to be absolutely sure
            value = value / len(neighbours)
        image[pixel[0], pixel[1]] = value
    return image
