import scipy
import numpy as np


def get_wavelength(energy):
    phir = energy * (1 + scipy.constants.e * energy / (2 * scipy.constants.m_e * scipy.constants.c**2))
    g = np.sqrt(2 * scipy.constants.m_e * scipy.constants.e * phir)
    k = g / scipy.constants.hbar
    wavelength = 2 * np.pi / k
    return wavelength  # in meters
