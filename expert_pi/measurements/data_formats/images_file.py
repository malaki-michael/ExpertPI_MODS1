import numpy as np

from stem_measurements.core import ImagesFileCore


class ImagesFile:
    """Class for caching images in a file."""

    def __init__(self, path: str, create_new: bool) -> None:
        """Create a new cache file.

        Args:
            path (str): path to the cache file.
        """
        self._file = ImagesFileCore(path, create_new)
        self._path = path

    def write(self, data: np.ndarray, batch_sizes: list[int]) -> None:
        """Write data to the cache file.

        Args:
            images (np.ndarray): compressed images as batch.
            batch_sizes (list[int]): list of sizes of each image in the batch.
        """
        self._file.write(data, batch_sizes)

    def read(self, indexes: list[int], output: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
        """Read data from the cache file as compressed batch.

        Args:
            indexes (list[int]): list of image indexes to read.
            output (np.ndarray | None, optional): output array to read images into. Defaults to None.

        Returns:
            tuple[np.ndarray, list[int]]: tuple of (compressed images, list of sizes of each image in the batch).
        """
        return self._file.read(indexes, output)

    def read_single(self, index: int, output: np.ndarray | None = None) -> np.ndarray:
        """Read data from the cache file as compressed single image.

        Args:
            index (int): index of the image to read.
            output (np.ndarray | None, optional): output array to read image into. Defaults to None.

        Returns:
            np.ndarray: compressed image.
        """
        return self._file.read_single(index, output)

    # def count(self) -> int:
    #     """Get the number of images in the cache file.

    #     Returns:
    #         int: number of images in the cache file.
    #     """
    #     return self._file.count()

    def get_buffer_size(self, indexes: list[int]) -> int:
        """Get the size of the buffer needed to hold all images in the batch for given indexes.

        Args:
            indexes (list[int]): list of image indexes to read.

        Returns:
            int: size of the buffer.
        """
        return self._file.get_buffer_size(indexes)

    # def reset(self) -> None:
    #     """Reset the cache file."""
    #     self._file = ImagesCacheCore(self._path)  # TODO: this is not thread safe, should be implemented in Rust
