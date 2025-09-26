from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.lib.recfunctions as rfn
from py4DSTEM.io.datastructure import PointList
from py4DSTEM.process.diffraction.WK_scattering_factors import compute_WK_factor
from py4DSTEM.process.utils import electron_wavelength_angstrom, single_atom_scatter
from scipy import linalg


@dataclass
class DynamicalMatrixCache:
    has_valid_cache: bool = False
    cached_U_gmh: np.array = None


# added orientation matrix instead of pure zone_axis_lattice

def generate_dynamical_diffraction_pattern(
        self,
        beams: PointList,
        thickness: Union[float, list, tuple, np.ndarray],
        orientation_matrix: np.ndarray = None,
        zone_axis_lattice: np.ndarray = None,
        zone_axis_cartesian: np.ndarray = None,
        foil_normal_lattice: np.ndarray = None,
        foil_normal_cartesian: np.ndarray = None,
        verbose: bool = False,
        always_return_list: bool = False,
        dynamical_matrix_cache: Optional[DynamicalMatrixCache] = None,
        return_complex: bool = False,
        return_eigenvectors: bool = False,
        return_Smatrix: bool = False,
) -> Union[PointList, List[PointList]]:
    """
    Generate a dynamical diffraction pattern (or thickness series of patterns)
    using the Bloch wave method.

    The beams to be included in the Bloch calculation must be pre-calculated
    and passed as a PointList containing at least (qx, qy, h, k, l) fields.

    If ``thickness`` is a single value, one new PointList will be returned.
    If ``thickness`` is a sequence of values, a list of PointLists will be returned,
        corresponding to each thickness value in the input.

    Frequent reference will be made to "Introduction to conventional transmission electron microscopy"
        by DeGraef, whose overall approach we follow here.

    Args:
        beams (PointList):              PointList from the kinematical diffraction generator
                                        which will define the beams included in the Bloch calculation
        thickness (float or list/array) thickness in Ångström to evaluate diffraction patterns at.
                                        The main Bloch calculation can be reused for multiple thicknesses
                                        without much overhead.
        zone_axis & foil_normal         Incident beam orientation and foil normal direction.
                                        Each can be specified in the Cartesian or crystallographic basis,
                                        using e.g. zone_axis_lattice or zone_axis_cartesian. These are
                                        internally parsed by Crystal.parse_orientation

    Less commonly used args:
        always_return_list (bool):      When True, the return is always a list of PointLists,
                                        even for a single thickness
        dynamical_matrix_cache:         (DyanmicalMatrixCache) Dataclass used for caching of the
                                        dynamical matrix. If the cached matrix does not exist, it is
                                        computed and stored. Subsequent calls will use the cached matrix
                                        for the off-diagonal components of the A matrix and overwrite
                                        the diagonal elements. This is used for CBED calculations.
        return_complex (bool):          When True, returns both the complex amplitude and intensity. Defaults to (False)
    Returns:
        bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, h, k, l]
            or
        [bragg_peaks,...] (PointList):   If thickness is a list/array, or always_return_list is True,
                                        a list of PointLists is returned.
        if return_complex = True:
            bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, amplitude, h, k, l]
                or
            [bragg_peaks,...] (PointList):   If thickness is a list/array, or always_return_list is True,
                                            a list of PointLists is returned.
        if return_Smatrix = True:
            [S_matrix, ...], psi_0:     Returns a list of S-matrices for each thickness (this is always a list),
                                        and the vector representing the incident plane wave. The beams of the
                                        S-matrix have the same order as in the input `beams`.

    """
    t0 = time()  # start timer for matrix setup

    n_beams = beams.data.shape[0]

    beam_g, beam_h = np.meshgrid(np.arange(n_beams), np.arange(n_beams))

    # Parse input orientations:
    if orientation_matrix is not None:
        zone_axis_rotation_matrix = orientation_matrix
    else:
        zone_axis_rotation_matrix = self.parse_orientation(
                zone_axis_lattice=zone_axis_lattice, zone_axis_cartesian=zone_axis_cartesian
        )
    if foil_normal_lattice is not None or foil_normal_cartesian is not None:
        foil_normal = self.parse_orientation(
                zone_axis_lattice=foil_normal_lattice,
                zone_axis_cartesian=foil_normal_cartesian,
        )
    else:
        foil_normal = zone_axis_rotation_matrix

    foil_normal = foil_normal[:, 2]

    # Note the difference in notation versus kinematic function:
    # k0 is the scalar magnitude of the wavevector, rather than
    # a vector along the zone axis.
    k0 = 1.0/electron_wavelength_angstrom(self.accel_voltage)

    ################################################################
    # Compute the reduced structure matrix \bar{A} in DeGraef 5.52 #
    ################################################################

    hkl = np.vstack((beams.data["h"], beams.data["k"], beams.data["l"])).T

    # Check if we have a cached dynamical matrix, which saves us from calculating the
    # off-diagonal elements when running this in a loop with the same zone axis
    if dynamical_matrix_cache is not None and dynamical_matrix_cache.has_valid_cache:
        U_gmh = dynamical_matrix_cache.cached_U_gmh
    else:
        # No cached matrix is available/desired, so calculate it:

        # get hkl indices of \vec{g} - \vec{h}
        g_minus_h = np.vstack(
                (
                    beams.data["h"][beam_g.ravel()] - beams.data["h"][beam_h.ravel()],
                    beams.data["k"][beam_g.ravel()] - beams.data["k"][beam_h.ravel()],
                    beams.data["l"][beam_g.ravel()] - beams.data["l"][beam_h.ravel()],
                )
        ).T

        # Get the structure factors for each nonzero element, and zero otherwise
        U_gmh = np.array(
                [
                    self.Ug_dict.get((gmh[0], gmh[1], gmh[2]), 0.0 + 0.0j)
                    for gmh in g_minus_h
                ],
                dtype=np.complex128,
        ).reshape(beam_g.shape)

    # If we are supposed to cache, but don't have one saved, save this one:
    if (
            dynamical_matrix_cache is not None
            and not dynamical_matrix_cache.has_valid_cache
    ):
        dynamical_matrix_cache.cached_U_gmh = U_gmh
        dynamical_matrix_cache.has_valid_cache = True

    if verbose:
        print(f"Bloch matrix has size {U_gmh.shape}")

    # Compute the diagonal entries of \hat{A}: 2 k_0 s_g [5.51]
    g = (hkl@self.lat_inv)@zone_axis_rotation_matrix
    sg = self.excitation_errors(
            g.T, foil_normal=-foil_normal@zone_axis_rotation_matrix
    )

    # Fill in the diagonal, completing the structure mattrx
    np.fill_diagonal(U_gmh, 2*k0*sg + 1.0j*np.imag(self.Ug_dict[(0, 0, 0)]))

    if verbose:
        print(f"Constructing the A matrix took {(time() - t0)*1000.:.3f} ms.")

    #############################################################################################
    # Compute eigen-decomposition of \hat{A} to yield C (the matrix containing the eigenvectors #
    # as its columns) and gamma (the reduced eigenvalues), as in DeGraef 5.52                   #
    #############################################################################################

    t0 = time()  # start timer for eigendecomposition

    v, C = linalg.eig(U_gmh)  # decompose!
    gamma_fac = 2.0*k0*zone_axis_rotation_matrix[:, 2]@foil_normal
    gamma = v/gamma_fac  # divide by 2 k_n

    # precompute the inverse of C
    C_inv = np.linalg.inv(C)

    if verbose:
        print(f"Decomposing the A matrix took {(time() - t0)*1000.:.3f} ms.")

    ##############################################################################################
    # Compute diffraction intensities by calculating exit wave \Psi in DeGraef 5.60, and collect #
    # values into PointLists                                                                     #
    ##############################################################################################

    t0 = time()

    psi_0 = np.zeros((n_beams,))
    psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0

    # calculate the diffraction intensities (and amplitudes) for each thichness matrix
    # I = |psi|^2 ; psi = C @ E(z) @ C^-1 @ psi_0, where E(z) is the thickness matrix
    if return_Smatrix:
        Smatrices = [
            C@np.diag(np.exp(2.0j*np.pi*z*gamma))@C_inv
            for z in np.atleast_1d(thickness)
        ]
        return (
            (Smatrices, psi_0, (C, C_inv, gamma, gamma_fac))
            if return_eigenvectors
            else (Smatrices, psi_0)
        )
    elif return_complex:
        # calculate the amplitudes
        amplitudes = [
            C@(np.exp(2.0j*np.pi*z*gamma)*(C_inv@psi_0))
            for z in np.atleast_1d(thickness)
        ]

        # Do this first to avoid handling structured array
        intensities = np.abs(amplitudes)**2

        # convert amplitudes as a structured array
        # do we want complex64 or complex 32.
        amplitudes = np.array(amplitudes, dtype=([("amplitude", "<c16")]))
    else:
        intensities = [
            np.abs(C@(np.exp(2.0j*np.pi*z*gamma)*(C_inv@psi_0)))**2
            for z in np.atleast_1d(thickness)
        ]

    # make new pointlists for each thickness case and copy intensities
    pls = []
    for i in range(len(intensities)):
        newpl = beams.copy()
        if return_complex:
            # overwrite the kinematical intensities with the dynamical intensities
            newpl.data["intensity"] = intensities[i]
            # merge amplitudes into the list
            newpl.data = rfn.merge_arrays(
                    (newpl.data, amplitudes[i]), asrecarray=False, flatten=True
            )
        else:
            newpl.data["intensity"] = intensities[i]
        pls.append(newpl)

    if verbose:
        print(f"Assembling outputs took {1000*(time() - t0):.3f} ms.")

    if len(pls) == 1 and not always_return_list:
        return pls[0] if not return_eigenvectors else (pls[0], gamma, C)
    else:
        return pls if not return_eigenvectors else (pls, gamma, C)
