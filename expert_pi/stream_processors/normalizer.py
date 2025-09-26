import numpy as np

from expert_pi.stream_processors import functions


class Normalizer:
    """Convert 16 bit -> 8bit using alpha beta from histogram."""

    def __init__(self, update_function=None, update_histogram_function=None):
        self.update_function = update_function
        self.update_histogram_function = update_histogram_function

        self.alpha = 1 / 255
        self.beta = 0

        self.histogram_amplify = 0
        self.histogram_enable = False

        self.shape = (2, 2)
        self.raw_image = np.zeros(self.shape, dtype=np.uint16)
        self.image_8b = np.zeros(self.shape, dtype=np.uint8)

        self.rectangle = None

        self.minimum_scan_id = 0

    def set_minimum_scan_id(self, minimum_scan_id):
        self.minimum_scan_id = minimum_scan_id

    def set_alpha_beta(self, alpha, beta, amplify=0.0):
        self.alpha = alpha
        self.beta = beta
        self.histogram_amplify = amplify

    def set_shape(self, shape):
        if shape != self.image_8b.shape and self.rectangle is None:
            self.shape = shape
            self.image_8b = np.zeros(shape, dtype=np.uint8)
            self.raw_image = np.zeros(shape, dtype=np.uint16)

    def set_rectangle(self, rectangle):
        """Rectangle: [i,j,i_count,j_count]."""
        self.rectangle = rectangle  # TODO check the size is possible

    def set_image(self, image, frame_index=0, scan_id=0):
        if image.shape != self.shape:
            self.set_shape(image.shape)
        self.set_chunk(image.ravel(), 0, frame_index, scan_id)

    def set_chunk(self, chunk, start, frame_index, scan_id):
        """Set chunk of data to the image.

        will delay previous thread by calculation
        chunk can be two dimensionals, but in that case start needs to be multiplied by shape[2]
        """
        chunk_size = len(chunk.ravel())
        if chunk_size == 0 or (scan_id is not None and scan_id < self.minimum_scan_id):
            return

        chunk_8b_flat = functions.calc_cast_numba(chunk.ravel(), self.alpha, self.beta)

        total = np.prod(self.shape)
        dim = 1
        if len(self.shape) == 3:
            dim = self.shape[2]
        if self.rectangle is None:
            # flat is slow, reshape will work the same and it is much faster
            self.raw_image.reshape(total)[start : start + chunk_size] = chunk.ravel()
            self.image_8b.reshape(total)[start : start + chunk_size] = chunk_8b_flat
        else:
            ir = self.rectangle
            # need to use flat here to write to array
            self.raw_image[ir[0] : ir[0] + ir[2], ir[1] : ir[1] + ir[3]].flat[start : start + chunk_size] = (
                chunk.ravel()
            )
            self.image_8b[ir[0] : ir[0] + ir[2], ir[1] : ir[1] + ir[3]].flat[start : start + chunk_size] = chunk_8b_flat
            rectangle = self.rectangle

        if self.update_function is not None:
            self.update_function(self.image_8b, frame_index, scan_id)

        if self.histogram_enable:
            if self.histogram_amplify == 0:
                histogram_data = functions.calc_hist_numba(self.raw_image.ravel())
            else:
                histogram_data = functions.calc_hist_numba_amplify(
                    self.raw_image.ravel(), int(2**self.histogram_amplify)
                )

            if self.update_histogram_function is not None:
                self.update_histogram_function(histogram_data, frame_index, scan_id)
