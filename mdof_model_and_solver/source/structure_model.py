# ===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek

        Structure model base class and derived classes for related structures

Author: mate.pentek@tum.de, anoop.kodakkal@tum.de, catharina.czech@tum.de, peter.kupas@tum.de

Note:   UPDATE: The script has been written using publicly available information and
        data, use accordingly. It has been written and tested with Python 2.7.9.
        Tested and works also with Python 3.4.3 (already see differences in print).
        Module dependencies (-> line 61-74):
            python
            numpy
            sympy
            matplotlib.pyplot

Created on:  22.11.2017
Last update: 23.11.2017
'''
# ===============================================================================

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from functools import partial
import math

CUST_MAGNITUDE = 4


def magnitude(x):
    # NOTE: ceil is supposed to be correct for positive values
    return int(math.ceil(math.log10(x)))


def map_lin_to_log(val, base=10**3):
    # it has to be defined with a min=0.0 and max=1.0
    # TODO: implment check
    return base**(1.0 - val)


def shift_normalize(val, base=10**3):
    # TODO: implment check
    # it has to be defined with min=0.0
    shift_val = map_lin_to_log(0.0, base)
    val -= shift_val
    # it has to be defined with max=1.0
    norm_val = map_lin_to_log(1.0, base) - shift_val
    val /= norm_val
    return val


# def shape_function_lin(val): return 1-val

# TODO: try to figure out a good relationship between exp and magnitude_difference
def shape_function_exp(val, exp=CUST_MAGNITUDE): return (1-val)**exp


class SimplifiedCantileverStructure(object):
    def __init__(self, m, b, k, nodal_coordinates, name, category):
        """
        Sets up the main structural properties (matrices)
        and includes the nodal coordinates (in height)
        as well as tags (strings) for naming and category.

        m, b, k matrices -> setup with the Dirichlet BC included

        nodal_coordinates dictionary with keys "x0", "y0" undeformed
        and "x", "y" for deformed (latter initially as None)
        -> origin / base node coordinate missing as Dirichlet BC is applied
        """

        # structural properties
        # mass matrix
        self.m = m
        # damping matrix
        self.b = b
        # stiffness matrix
        self.k = k

        # geometric properties
        self.nodal_coordinates = nodal_coordinates

        # a custom given name
        self.name = name
        # a chosen category - "MDoF2DMixed"
        if category in ['MDoF2DMixed', 'MDoF3DMixed']:
            self.category = category
        else:
            err_msg = 'The requested category \"' + category + \
                '\" is not supported in SimplifiedCantileverStructure\n'
            err_msg += 'Available options are: \"MDoF2DMixed\", \"MDoF3DMixed\"'
            raise Exception(err_msg)


class MDoFMixed2DModel(SimplifiedCantileverStructure):
    """
    A multi-degree-of-freedom MDoF model assuming
    bending-type deformations using the Euler-Bernoulli
    beam theory.

    ATTENTION:
    For this model a homogenous distribution of mass,
    stiffness and damping is a premise. For other cases
    this model is not adequate and changes need to be done.

    """

    def __init__(self,
                 rho=1.,
                 area=1.,
                 target_freq=1.,
                 target_mode=1,
                 zeta=0.05,
                 level_height=1,
                 num_of_levels=10,
                 gamma=0.5,
                 name="DefaultMDoF2DMixedModel"):

        self.gamma = gamma

        self.dofs_per_node = 2
        self.nodes_per_level = 2

        m = self._calculate_mass(rho, area, level_height, num_of_levels)
        k = self._calculate_stiffness(
            m, level_height, num_of_levels, target_freq, target_mode)
        b = self._calculate_damping(m, k, zeta)

        height_coordinates = self._get_nodal_coordinates(
            level_height, num_of_levels)

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoF2DMixed')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1, num_of_levels+1)
        return nodal_coordinates

    def _calculate_mass(self, rho, area, level_height, num_of_levels, is_lumped=False):
        """
        Getting the consistant mass matrix based on analytical integration
        """
        # mass values for one level
        length = level_height
        m_const = rho * area * length

        if not is_lumped:
            # case of consistent mass matrix
            m_beam_elem = 1/420 * np.array([[156, 22*length, 54,   -13*length],
                                            [22*length, 4*length**2,
                                                13*length, -3*length**2],
                                            [54, 13*length, 156, -22*length],
                                            [-13*length, -3*length**2, -22*length, 4*length**2]])

            # NOTE: for now 1/175 on diagonal for rotational inertia otherwise problems with eigenvalue solve
            # 1/175 seems to be the lowest value for pure shear to still deliver the correct eigenfreq
            m_shear_elem = 1/6 * np.array([[2,  0,  1,  0],
                                           [0,  1/175, 0,  0],
                                           [1,  0,  2,  0],
                                           [0,  0,  0,  1/175]])

        else:
            # case of lumped mass matrix
            alpha = 1/50  # generally between 0 and 1/50
            m_beam_elem = np.array([[1/2, 0,                 0,   0],
                                    [0,   alpha * length**2, 0,   0],
                                    [0,   0,                 1/2, 0],
                                    [0,   0,                 0,   alpha * length ** 2]])

            # NOTE: for now 1/2500 on diagonal for rotational inertia otherwise problems with eigenvalue solve
            # 1/2500 seems to be the lowest value for pure shear to still deliver the correct eigenfreq
            m_shear_elem = np.array([[1/2, 0,   0,   0],
                                     [0,   1/2500,   0,   0],
                                     [0,   0,   1/2, 0],
                                     [0,   0,   0,   1/2500]])

        # global mass matrix initialization with zeros
        m_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))

        # shape function values for superposition of bending and shear
        sf_b_val = (1-shape_function_exp(shift_normalize(
                    map_lin_to_log(self.gamma))))
        sf_s_val = shape_function_exp(shift_normalize(
            map_lin_to_log(self.gamma)))

        # fill global mass matrix entries
        for i in range(num_of_levels):
            m_temp = np.zeros(
                (2 * num_of_levels + 2, 2 * num_of_levels + 2))
            m_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = sf_b_val * \
                m_beam_elem + sf_s_val * m_shear_elem
            m_glob += m_const * m_temp

        # remove the fixed degrees of freedom
        # at first node all dofs are considered fixed for the cantilever beam
        # generate a list of indices 0, 1, 2,...self.dofs_per_node
        # go through it in reverse order to keep the numbering intact
        for dof in range(self.dofs_per_node)[::-1]:
            # delete corresponding row
            m_glob = np.delete(m_glob, dof, axis=0)
            # delet corresponding column
            m_glob = np.delete(m_glob, dof, axis=1)

        # return stiffness matrix
        return m_glob

    def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
        """
        Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all
        the elements and the value is calculated using an optimization (or "tuning")
        for a target frequency of a target mode.
        """
        print("Calculating stiffness k in MDoFMixedModel derived class \n")

        # setup k_scalar_guess as the input for the standard k for a shear-type
        # MDoF
        # guess stiffness values for shear and bending
        # as initial value for optimization

        k_scalar_g = 1000.

        # using partial to fix some parameters for the
        optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar,
                                       m,
                                       level_height,
                                       num_of_levels,
                                       target_freq,
                                       target_mode)

        print("Optimization for the target k matrix in MDoFMixedModel \n")
        minimization_result = minimize(optimizable_function,
                                       k_scalar_g, method='Powell',
                                       options={'disp': True})

        # returning only one value!
        k_scalar_opt = minimization_result.x
        print("K scalar optimized: ", k_scalar_opt, '\n')

        return self._assemble_k(level_height, num_of_levels, k_scalar_opt)

    def _assemble_k(self, level_height, num_of_levels, k_scalar, magnitude_difference=CUST_MAGNITUDE, beta=0.0):

        # pure bending
        # gamma = 1.0
        # pure shear
        # gamma = 0.0
        """
        For the MDoFMixed model stiffness distribution according to beam theory is assumed
        the stiffness matrix is asembled with the k_scalar calculated.
        """

        # element length / level height
        length = level_height

        # beam stiffness
        k_beam = k_scalar

        # shear (spring) stiffness
        k_shear = k_scalar

        # stifness values for one level
        # TODO: check as it seems Timoschenko beam needs some additional off-diagonal terms to be affected by beta
        # similar as in 3D
        k_beam_elem = 1/(1+beta) * np.array([[12,     6 * length,         -12,     6 * length],
                                             [6 * length, (4+beta) * length **
                                              2, -6 * length, 2 * length ** 2],
                                             [-12,    -6 * length,
                                                 12,    -6 * length],
                                             [6 * length, 2 * length ** 2, -6 * length, (4+beta) * length ** 2]])

        shear_rotation_magnitude = magnitude(
            (4+beta) * length ** 2) + magnitude_difference

        k_shear_elem = np.array([[12, 0, -12, 0],
                                 [0, (4+beta) * length **
                                  2 * 10**shear_rotation_magnitude, 0, 0],
                                 [-12, 0, 12, 0],
                                 [0, 0, 0, (4+beta) * length **
                                  2 * 10**shear_rotation_magnitude]])

        # global stiffness matrix initialization with zeros
        k_glob = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                           (num_of_levels+1) * self.dofs_per_node))

        # shape function values for superposition of bending and shear
        sf_b_val = (1-shape_function_exp(shift_normalize(
                    map_lin_to_log(self.gamma))))
        sf_s_val = shape_function_exp(shift_normalize(
            map_lin_to_log(self.gamma)))

        print('Gamma bending 1.0')
        print('Gamma shear 0.0')
        print('Gamma prescribed ', self.gamma, '\n')

        print('S(hape) f(unction) values for')
        print('Shear ', sf_s_val)
        print('Bend ', sf_b_val)
        print('\n')

        # fill global stiffness matix entries
        for i in range(num_of_levels):
            k_temp = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                               (num_of_levels+1) * self.dofs_per_node))

            # beam part
            k_temp[self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level,
                   self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level] = sf_b_val * k_beam * k_beam_elem
            # shear part

            k_temp[self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level,
                   self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level] += sf_s_val * k_shear * k_shear_elem

            k_glob += k_temp

        # remove the fixed degrees of freedom
        # at first node all dofs are considered fixed for the cantilever beam
        # generate a list of indices 0, 1, 2,...self.dofs_per_node
        # go through it in reverse order to keep the numbering intact
        for dof in range(self.dofs_per_node)[::-1]:
            # delete corresponding row
            k_glob = np.delete(k_glob, dof, axis=0)
            # delet corresponding column
            k_glob = np.delete(k_glob, dof, axis=1)

        # return stiffness matrix
        return k_glob

    def _calculate_damping(self, m, k, zeta):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """
        print("Calculating damping b in MDoFMixedModel derived class \n")
        mode_i = 0
        mode_j = 1
        zeta_i = zeta
        zeta_j = zeta

        # TODO: try to avoid this code duplication

        # # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        # print(" If mass is positive definite should return True")
        # print(np.all(np.linalg.eigvals(m) > 0))
        # wait = input("check...")

        # raw results
        eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        #

        a = np.linalg.solve(0.5 *
                            np.array(
                                [[1 / eig_values[eig_freqs_sorted_indices[mode_i]],
                                  eig_values[
                                  eig_freqs_sorted_indices[mode_i]]],
                                    [1 / eig_values[eig_freqs_sorted_indices[mode_j]],
                                     eig_values[
                                     eig_freqs_sorted_indices[
                                         mode_j]]]]),
                            [zeta_i, zeta_j])
        return a[0] * m + a[1] * k

    # k_scalar, k_beam):
    def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
        k = self._assemble_k(level_height, num_of_levels,
                             k_scalar)

        # TODO: try to avoid this code duplication
        # raw results
        eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        #

        current_target_freq = eig_freqs[eig_freqs_sorted_indices[target_mode-1]]

        return (target_freq - current_target_freq) ** 2 / target_freq**2

    def _load_distribution(self, level_height, num_of_levels, wind_forces):

        # Nodal distribution the applis load on a beam model
        def nodal_force(force):
            F = [force / 2, force * level_height / 12,
                 force / 2, -force * level_height / 12]
            return F

        nodal_load = list(map(nodal_force, wind_forces))

        load = np.zeros(2 * num_of_levels + 2)

        for i in range(num_of_levels):
            load_temp = np.zeros(2 * num_of_levels + 2)
            load_temp[2 * i:2 * i + 4] = nodal_load[i]
            load += load_temp

        # remove the fixed degrees of freedom
        rdof = [1, 0]
        for dof in rdof:
            load = np.delete(load, dof, axis=0)

        return load

# NOTE: for now only a copy of the 2D - WIP 

class MDoFMixed3DModel(SimplifiedCantileverStructure):
    """
    A multi-degree-of-freedom MDoF model assuming
    bending-type deformations using the Euler-Bernoulli
    beam theory.

    ATTENTION:
    For this model a homogenous distribution of mass,
    stiffness and damping is a premise. For other cases
    this model is not adequate and changes need to be done.

    """

    def __init__(self,
                 rho=1.,
                 area=1.,
                 target_freq=1.,
                 target_mode=1,
                 zeta=0.05,
                 level_height=1,
                 num_of_levels=10,
                 gamma=0.5,
                 name="DefaultMDoF3DMixedModel"):

        self.gamma = gamma

        self.dofs_per_node = 6
        self.nodes_per_level = 2

        m = self._calculate_mass(rho, area, level_height, num_of_levels)
        k = self._calculate_stiffness(
            m, level_height, num_of_levels, target_freq, target_mode)
        b = self._calculate_damping(m, k, zeta)

        height_coordinates = self._get_nodal_coordinates(
            level_height, num_of_levels)

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoF2DMixed')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1, num_of_levels+1)
        return nodal_coordinates

    def _calculate_mass(self, rho, area, level_height, num_of_levels):
        """
        Getting the consistant mass matrix based on analytical integration

        USING the consistent mass formulation
        """
        # mass values for one level
        length = level_height
        m_const = rho * area * length

        # mass values for one level

        # define component-wise to have enable better control for various optimization parameters

        # axial inertia - along axis x - here marked as x
        m_x = m_const / 6.0 
        m_x_11 = 2.
        m_x_12 = 1.
        m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                 [m_x_12, m_x_11]])
        # torsion inertia - around axis x - here marked as alpha - a
        # NOTE: assuming torsional inertia to be x100 greates than translation
        m_a = m_const / 6.0 * 100 
        m_a_11 = 2
        m_a_12 = 1
        m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                 [m_a_12, m_a_11]])
        # bending - inertia along axis y, rotations around axis z - here marked as beta - b
        m_yb = m_const / 420        
        #
        m_yb_11 = 156.
        m_yb_12 = 22*length
        m_yb_13 = 54.
        m_yb_14 = -13.*length
        #
        m_yb_22 = 4 * length **2
        m_yb_23 = 13. * length
        m_yb_24 = -3 * length ** 2
        #
        m_yb_33 = 156.
        m_yb_34 = -22. * length
        #
        m_yb_44 = 4 * length ** 2
        #
        m_el_yb = m_yb * np.array([[m_yb_11, m_yb_12, m_yb_13, m_yb_14],
                                   [m_yb_12, m_yb_22, m_yb_23, m_yb_24],
                                   [m_yb_13, m_yb_23, m_yb_33, m_yb_34],
                                   [m_yb_14, m_yb_24, m_yb_34, m_yb_44]])

        # bending - inertia along axis z, rotations around axis y - here marked as gamma - g
        # NOTE: some signs swapped as for stiffness - is this correct?
        m_zg = m_const / 420
        #
        m_zg_11 = 156.
        m_zg_12 = (-1) * 22*length
        m_zg_13 = 54.
        m_zg_14 = (-1) * 13.*length
        #
        m_zg_22 = 4. * length **2
        m_zg_23 = (-1) * 13. * length
        m_zg_24 = -3 * length ** 2
        #
        m_zg_33 = 156.
        m_zg_34 = (-1) * 22. * length
        #
        m_zg_44 = 4 * length ** 2
        #
        m_el_zg = m_zg * np.array([[m_zg_11, m_zg_12, m_zg_13, m_zg_14],
                                   [m_zg_12, m_zg_22, m_zg_23, m_zg_24],
                                   [m_zg_13, m_zg_23, m_zg_33, m_zg_34],
                                   [m_zg_14, m_zg_24, m_zg_34, m_zg_44]])

        # assemble all components
        m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0.,                 m_el_x[0][1], 0., 0., 0., 0., 0.],
                         [0., m_el_yb[0][0], 0., 0., 0., m_el_yb[0][1],     0., m_el_yb[0][2], 0., 0., 0., m_el_yb[0][3]],
                         [0., 0., m_el_zg[0][0], 0., m_el_zg[0][1], 0.,     0., 0., m_el_zg[0][2], 0., m_el_zg[0][3], 0.],
                         [0., 0., 0., m_el_a[0][0], 0., 0.,                 0., 0., 0., m_el_a[0][1], 0., 0.],
                         [0., 0., m_el_zg[0][1], 0., m_el_zg[1][1], 0.,     0., 0., m_el_zg[1][2], 0., m_el_zg[1][3], 0.],
                         [0., m_el_yb[0][1], 0., 0., 0., m_el_yb[1][1],     0., m_el_yb[1][2], 0., 0., 0., m_el_yb[1][3]],
                         
                         [m_el_x[1][0], 0., 0., 0., 0., 0.,                 m_el_x[1][1], 0., 0., 0., 0., 0.],
                         [0., m_el_yb[0][2], 0., 0., 0., m_el_yb[1][2],     0., m_el_yb[2][2], 0., 0., 0., m_el_yb[2][3]],
                         [0., 0., m_el_zg[0][2], 0., m_el_zg[1][2], 0.,     0., 0., m_el_zg[2][2], 0., m_el_zg[2][3], 0.],
                         [0., 0., 0., m_el_a[1][0], 0., 0.,                 0., 0., 0., m_el_a[1][1], 0., 0.],
                         [0., 0., m_el_zg[0][3], 0., m_el_zg[1][3], 0.,     0., 0., m_el_zg[2][3], 0., m_el_zg[3][3], 0.],
                         [0., m_el_yb[0][3], 0., 0., 0., m_el_yb[1][3],     0., m_el_yb[2][3], 0., 0., 0., m_el_yb[3][3]]])

        # global mass matrix initialization with zeros
        m_glob = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                           (num_of_levels+1) * self.dofs_per_node))


        # fill global mass matix entries
        for i in range(num_of_levels):
            m_temp = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                               (num_of_levels+1) * self.dofs_per_node))

            # beam part
            m_temp[self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level,
                   self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level] = m_el

            m_glob += m_temp

        # remove the fixed degrees of freedom
        # at first node all dofs are considered fixed for the cantilever beam
        # generate a list of indices 0, 1, 2,...self.dofs_per_node
        # go through it in reverse order to keep the numbering intact
        for dof in range(self.dofs_per_node)[::-1]:
            # delete corresponding row
            m_glob = np.delete(m_glob, dof, axis=0)
            # delet corresponding column
            m_glob = np.delete(m_glob, dof, axis=1)

        # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        print(" If mass global is positive definite should return True")
        print(np.all(np.linalg.eigvals(m_glob) > 0))
        wait = input("check...")

        # return mass matrix
        return m_glob


    def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
        """
        Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all
        the elements and the value is calculated using an optimization (or "tuning")
        for a target frequency of a target mode.
        """
        print("Calculating stiffness k in MDoFMixedModel derived class \n")

        # setup k_scalar_guess as the input for the standard k for a shear-type
        # MDoF
        # guess stiffness values for shear and bending
        # as initial value for optimization

        # k_scalar_g = 1000.

        # # using partial to fix some parameters for the
        # optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar,
        #                                m,
        #                                level_height,
        #                                num_of_levels,
        #                                target_freq,
        #                                target_mode)

        # print("Optimization for the target k matrix in MDoFMixedModel \n")
        # minimization_result = minimize(optimizable_function,
        #                                k_scalar_g, method='Powell',
        #                                options={'disp': True})

        # # returning only one value!
        # k_scalar_opt = minimization_result.x
        # print("K scalar optimized: ", k_scalar_opt, '\n')

        return self._assemble_k(level_height, num_of_levels)

    def _assemble_k(self, level_height, num_of_levels):
        length = level_height

        # stifness values for one level

        # define component-wise to have enable better control for various optimization parameters

        # axial stiffness - along axis x - here marked as x
        k_x = 1500. # E*A/l
        k_x_11 = 1.0
        k_x_12 = -1.0
        k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                 [k_x_12, k_x_11]])
        # torsion stiffness - around axis x - here marked as alpha - a
        # NOTE: assuming that torsional stiffness is 100 longitudnal one, probably incorrect
        k_a = 150 * 100.  # G*K/l
        k_a_11 = 1.0
        k_a_12 = -1.0
        k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                 [k_a_12, k_a_11]])
        # bending - displacement along axis y, rotations around axis z - here marked as beta - b
        beta_yb = 0.001
        k_yb = 500 #E * I_z / (1+beta_yb)**2 / h**2
        #
        k_yb_11 = 12.
        k_yb_12 = 6*length
        k_yb_13 = -12.
        k_yb_14 = 6.*length
        #
        k_yb_22 = (4.+beta_yb) * length **2
        k_yb_23 = -6. * length
        k_yb_24 = (2-beta_yb) * length ** 2
        #
        k_yb_33 = 12.
        k_yb_34 = -6. * length
        #
        k_yb_44 = (4+beta_yb) * length ** 2
        #
        k_el_yb = k_yb * np.array([[k_yb_11, k_yb_12, k_yb_13, k_yb_14],
                                   [k_yb_12, k_yb_22, k_yb_23, k_yb_24],
                                   [k_yb_13, k_yb_23, k_yb_33, k_yb_34],
                                   [k_yb_14, k_yb_24, k_yb_34, k_yb_44]])

        # bending - displacement along axis z, rotations around axis y - here marked as gamma - g
        beta_zg = 0.0025
        k_zg = 500 #E * I_y / (1+beta_zg)**2 / h**2
        #
        k_zg_11 = 12.
        k_zg_12 = -6*length
        k_zg_13 = -12.
        k_zg_14 = -6.*length
        #
        k_zg_22 = (4.+beta_zg) * length **2
        k_zg_23 = 6. * length
        k_zg_24 = (2-beta_zg) * length ** 2
        #
        k_zg_33 = 12.
        k_zg_34 = 6. * length
        #
        k_zg_44 = (4+beta_zg) * length ** 2
        #
        k_el_zg = k_zg * np.array([[k_zg_11, k_zg_12, k_zg_13, k_zg_14],
                                   [k_zg_12, k_zg_22, k_zg_23, k_zg_24],
                                   [k_zg_13, k_zg_23, k_zg_33, k_zg_34],
                                   [k_zg_14, k_zg_24, k_zg_34, k_zg_44]])

        # assemble all components
        k_el = np.array([[k_el_x[0][0], 0., 0., 0., 0., 0.,                 k_el_x[0][1], 0., 0., 0., 0., 0.],
                         [0., k_el_yb[0][0], 0., 0., 0., k_el_yb[0][1],     0., k_el_yb[0][2], 0., 0., 0., k_el_yb[0][3]],
                         [0., 0., k_el_zg[0][0], 0., k_el_zg[0][1], 0.,     0., 0., k_el_zg[0][2], 0., k_el_zg[0][3], 0.],
                         [0., 0., 0., k_el_a[0][0], 0., 0.,                 0., 0., 0., k_el_a[0][1], 0., 0.],
                         [0., 0., k_el_zg[0][1], 0., k_el_zg[1][1], 0.,     0., 0., k_el_zg[1][2], 0., k_el_zg[1][3], 0.],
                         [0., k_el_yb[0][1], 0., 0., 0., k_el_yb[1][1],     0., k_el_yb[1][2], 0., 0., 0., k_el_yb[1][3]],
                         
                         [k_el_x[1][0], 0., 0., 0., 0., 0.,                 k_el_x[1][1], 0., 0., 0., 0., 0.],
                         [0., k_el_yb[0][2], 0., 0., 0., k_el_yb[1][2],     0., k_el_yb[2][2], 0., 0., 0., k_el_yb[2][3]],
                         [0., 0., k_el_zg[0][2], 0., k_el_zg[1][2], 0.,     0., 0., k_el_zg[2][2], 0., k_el_zg[2][3], 0.],
                         [0., 0., 0., k_el_a[1][0], 0., 0.,                 0., 0., 0., k_el_a[1][1], 0., 0.],
                         [0., 0., k_el_zg[0][3], 0., k_el_zg[1][3], 0.,     0., 0., k_el_zg[2][3], 0., k_el_zg[3][3], 0.],
                         [0., k_el_yb[0][3], 0., 0., 0., k_el_yb[1][3],     0., k_el_yb[2][3], 0., 0., 0., k_el_yb[3][3]]])

        # global stiffness matrix initialization with zeros
        k_glob = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                           (num_of_levels+1) * self.dofs_per_node))


        # fill global stiffness matix entries
        for i in range(num_of_levels):
            k_temp = np.zeros(((num_of_levels+1) * self.dofs_per_node,
                               (num_of_levels+1) * self.dofs_per_node))

            # beam part
            k_temp[self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level,
                   self.dofs_per_node * i: self.dofs_per_node * i + self.dofs_per_node * self.nodes_per_level] = k_el

            k_glob += k_temp

        # remove the fixed degrees of freedom
        # at first node all dofs are considered fixed for the cantilever beam
        # generate a list of indices 0, 1, 2,...self.dofs_per_node
        # go through it in reverse order to keep the numbering intact
        for dof in range(self.dofs_per_node)[::-1]:
            # delete corresponding row
            k_glob = np.delete(k_glob, dof, axis=0)
            # delet corresponding column
            k_glob = np.delete(k_glob, dof, axis=1)

        # return stiffness matrix
        return k_glob

    def _calculate_damping(self, m, k, zeta):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """
        print("Calculating damping b in MDoFMixedModel derived class \n")
        mode_i = 0
        mode_j = 1
        zeta_i = zeta
        zeta_j = zeta

        # TODO: try to avoid this code duplication
        # raw results

        # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        print(" If mass is positive definite should return True")
        print(np.all(np.linalg.eigvals(m) > 0))
        wait = input("check...")

        eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        #

        a = np.linalg.solve(0.5 *
                            np.array(
                                [[1 / eig_values[eig_freqs_sorted_indices[mode_i]],
                                  eig_values[
                                  eig_freqs_sorted_indices[mode_i]]],
                                    [1 / eig_values[eig_freqs_sorted_indices[mode_j]],
                                     eig_values[
                                     eig_freqs_sorted_indices[
                                         mode_j]]]]),
                            [zeta_i, zeta_j])
        return a[0] * m + a[1] * k

    # k_scalar, k_beam):
    def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
        k = self._assemble_k(level_height, num_of_levels)

        # TODO: try to avoid this code duplication
        # raw results
        eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        #

        current_target_freq = eig_freqs[eig_freqs_sorted_indices[target_mode-1]]

        return (target_freq - current_target_freq) ** 2 / target_freq**2

    def _load_distribution(self, level_height, num_of_levels, wind_forces):

        # Nodal distribution the applis load on a beam model
        def nodal_force(force):
            F = [force / 2, force * level_height / 12,
                 force / 2, -force * level_height / 12]
            return F

        nodal_load = list(map(nodal_force, wind_forces))

        load = np.zeros(2 * num_of_levels + 2)

        for i in range(num_of_levels):
            load_temp = np.zeros(2 * num_of_levels + 2)
            load_temp[2 * i:2 * i + 4] = nodal_load[i]
            load += load_temp

        # remove the fixed degrees of freedom
        rdof = [1, 0]
        for dof in rdof:
            load = np.delete(load, dof, axis=0)

        return load