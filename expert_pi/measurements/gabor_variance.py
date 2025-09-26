import dataclasses
import math

import cv2
import numpy as np

try:
    import torch
except (ImportError, ModuleNotFoundError):
    # import warnings
    # warnings.warn("GaborKernel requires torch to be installed")
    torch = None


@dataclasses.dataclass()
class GaborKernelSettings:
    num_orientations: int = 32
    num_scales: int = 5
    kernel_size: int = 15
    wavelength: int = 8
    sigma: float | None = None

    def __post_init__(self):
        if self.sigma is None:
            self.sigma = math.pi / self.wavelength * 4


class GaborKernel:
    def __init__(
        self,
        gabor_kernel_settings: GaborKernelSettings | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("GaborKernel requires torch to be installed")
        if gabor_kernel_settings is not None:
            self.gabor_filters = self._get_gabor(gabor_kernel_settings)
        else:
            self.gabor_filters = self._get_gabor(GaborKernelSettings())

    def __call__(self, image: np.ndarray, device: str = "cpu"):
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.mean(axis=2)
            image_resized = image.copy()
            image_resized = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0)
            image_resized = image_resized / image_resized.max()
            features = torch.nn.functional.conv2d(
                image_resized.to(device), self.gabor_filters.to(device), padding=self.gabor_filters.shape[-1] // 2
            )
            return features.squeeze(0)

    def variance(self, image: np.ndarray, device: str = "cpu"):
        return float(self(image, device).var((1, 2)).mean().item())

    @staticmethod
    def _get_gabor(gabor_kernel_settings: GaborKernelSettings):
        theta = torch.linspace(0, math.pi, gabor_kernel_settings.num_orientations)  # orientation
        phi = torch.linspace(0, math.pi / 2, gabor_kernel_settings.num_scales)  # scale

        # create a tensor to hold the Gabor filters
        gabor_filters = torch.zeros(
            gabor_kernel_settings.num_orientations * gabor_kernel_settings.num_scales,
            1,
            gabor_kernel_settings.kernel_size,
            gabor_kernel_settings.kernel_size,
        )

        # fill the tensor with Gabor filters
        for i in range(gabor_kernel_settings.num_orientations):
            for j in range(gabor_kernel_settings.num_scales):
                kernel = cv2.getGaborKernel(
                    (gabor_kernel_settings.kernel_size, gabor_kernel_settings.kernel_size),
                    gabor_kernel_settings.sigma,
                    float(theta[i]),
                    gabor_kernel_settings.wavelength,
                    float(phi[j]),
                    ktype=cv2.CV_32F,
                )
                gabor_filters[i * gabor_kernel_settings.num_scales + j, 0, :, :] = torch.from_numpy(kernel)
        return gabor_filters


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = cv2.imread("../TEM_alignment/data/precession_deprecession/04-04-2023_10-15-45.tif", cv2.IMREAD_UNCHANGED)
    gabor = GaborKernel(kernel_size=15, wavelength=8, num_scales=10)
    img2 = cv2.GaussianBlur(image, (0, 0), 3, 3)
    grad_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, scale=8, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, scale=8, borderType=cv2.BORDER_DEFAULT)
    img = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    plt.subplot(1, 2, 1)
    plt.imshow(gabor(image).permute(1, 2, 0).detach().abs().cpu().numpy().max(axis=2))
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()
