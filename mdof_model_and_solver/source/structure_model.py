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

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoF3DMixed')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1, num_of_levels+1)
        return nodal_coordinates

    def _calculate_mass(self, rho, area, level_height, num_of_levels):
        """
        Getting the consistant mass matrix based on analytical integration

        USING the consistent mass formulation
        """

        # mass values for one level

        #
        # VERSION 1
        #
        # NOTE: checking out alternative implementation
        # according to https://mediatum.ub.tum.de/doc/1072355/file.pdf
        # description and implementation seems correct
        # NOTE: find out where the formulation for the mass comes from, stiffness seems standard

        t = 45.0 # t: beam thickness (y) [m]
        h = 30.0 # h: beam height (z) [m]
        rho = 160.0 # density of steel [kg/mˆ3]
        E = 2.861e8 # E: Young's modulus of steel [N/mˆ2]
        nu = 3/10 # nu: Poisson's ratio

        G = E/2/(1+nu) # G: Shear modulus [N/mˆ2]
        l = level_height # l: beam element length
        A = t*h # beam area [mˆ2]
        ASy = 5/6*A    
        ASz = 5/6*A # effective area of shear
        Iy = 1/12*h**3*t    
        Iz = 1/12*t**3*h # second moments of area [mˆ4]
        Ip = 1/12*t*h*(h**2+t**2) # polar moment of inertia [mˆ4]
        It = min([h,t])**3 *max([h,t])/7 # torsion constant [mˆ4]
        Py = 12*E*Iz/(G*ASy*l**2) #
        Pz = 12*E*Iy/(G*ASz*l**2) #

        M11 = np.zeros((6,6))
        M11[0][0] = 1/3
        M11[1][1] = 13/35 + 6*Iz/(5*A*l**2)
        M11[2][2] = 13/35 + 6*Iy/(5*A*l**2)
        M11[3][3] = Ip/(3*A)
        M11[4][4] = l**2/105 + 2*Iy/(15*A)
        M11[5][5] = l**2/105 + 2*Iz/(15*A)
        M11[5][1] = 11*l/210 + Iz/(10*A*l)
        M11[1][5] = M11[5][1]
        M11[4][2] =-11*l/210-Iy/(10*A*l)
        M11[2][4] = M11[4][2]

        M22 = -M11 + 2*np.diag(np.diag(M11))

        M21 = np.zeros((6,6))
        M21[0][0] = 1/6
        M21[1][1] = 9/70-6*Iz/(5*A*l**2)
        M21[2][2] = 9/70-6*Iy/(5*A*l**2)
        M21[3][3] = Ip/(6*A)
        M21[4][4] =-l**2/140-Iy/(30*A)
        M21[5][5] =-l**2/140-Iz/(30*A)
        M21[5][1] =-13*l/420 + Iz/(10*A*l)
        M21[1][5] =-M21[5][1]
        M21[4][2] = 13*l/420-Iy/(10*A*l)
        M21[2][4] =-M21[4][2]

        # mass values for one level
        length = level_height
        m_const = rho * A * length

        m_el = np.zeros((2*6,2*6))
        # upper left
        m_el[0:6,0:6] += m_const * M11
        # lower left
        m_el[6:12,0:6] += m_const * M21
        # upper right
        m_el[0:6,6:12] += m_const * np.transpose(M21)
        # lower right
        m_el[6:12,6:12] += m_const * M22

        version1_m_el = m_el
        # print("\n VERSION1 - m_el")
        # print(np.array2string(version1_m_el, precision=3, separator=',', suppress_small=True))

        #
        # VERSION 2
        #
        # NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
        # seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
        # implemented mass matrices similar to the stiffness one

        # define component-wise to have enable better control for various optimization parameters

        # axial inertia - along axis x - here marked as x
        m_x = m_const / 6.0 
        m_x_11 = 2.
        m_x_12 = 1.
        m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                 [m_x_12, m_x_11]])
        # torsion inertia - around axis x - here marked as alpha - a
        m_a = m_const * Ip/A / 6.0 
        m_a_11 = 2
        m_a_12 = 1
        m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                 [m_a_12, m_a_11]])
        # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
        m_yg = m_const / 420        
        #
        m_yg_11 = 156.
        m_yg_12 = 22*length
        m_yg_13 = 54.
        m_yg_14 = -13.*length
        #
        m_yg_22 = 4 * length **2
        m_yg_23 = 13. * length
        m_yg_24 = -3 * length ** 2
        #
        m_yg_33 = 156.
        m_yg_34 = -22. * length
        #
        m_yg_44 = 4 * length ** 2
        #
        m_el_yg = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                   [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                   [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                   [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

        # bending - inertia along axis z, rotations around axis y - here marked as beta - b
        m_zb = m_const / 420
        #
        m_zb_11 = 156.
        m_zb_12 = -22*length
        m_zb_13 = 54.
        m_zb_14 = 13.*length
        #
        m_zb_22 = 4. * length **2
        m_zb_23 = -13. * length
        m_zb_24 = -3 * length ** 2
        #
        m_zb_33 = 156.
        m_zb_34 = 22. * length
        #
        m_zb_44 = 4 * length ** 2
        #
        m_el_zb = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                   [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                   [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                   [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])

        # assemble all components
        m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0.,                 m_el_x[0][1], 0., 0., 0., 0., 0.],
                         [0., m_el_yg[0][0], 0., 0., 0., m_el_yg[0][1],     0., m_el_yg[0][2], 0., 0., 0., m_el_yg[0][3]],
                         [0., 0., m_el_zb[0][0], 0., m_el_zb[0][1], 0.,     0., 0., m_el_zb[0][2], 0., m_el_zb[0][3], 0.],
                         [0., 0., 0., m_el_a[0][0], 0., 0.,                 0., 0., 0., m_el_a[0][1], 0., 0.],
                         [0., 0., m_el_zb[0][1], 0., m_el_zb[1][1], 0.,     0., 0., m_el_zb[1][2], 0., m_el_zb[1][3], 0.],
                         [0., m_el_yg[0][1], 0., 0., 0., m_el_yg[1][1],     0., m_el_yg[1][2], 0., 0., 0., m_el_yg[1][3]],
                         
                         [m_el_x[1][0], 0., 0., 0., 0., 0.,                 m_el_x[1][1], 0., 0., 0., 0., 0.],
                         [0., m_el_yg[0][2], 0., 0., 0., m_el_yg[1][2],     0., m_el_yg[2][2], 0., 0., 0., m_el_yg[2][3]],
                         [0., 0., m_el_zb[0][2], 0., m_el_zb[1][2], 0.,     0., 0., m_el_zb[2][2], 0., m_el_zb[2][3], 0.],
                         [0., 0., 0., m_el_a[1][0], 0., 0.,                 0., 0., 0., m_el_a[1][1], 0., 0.],
                         [0., 0., m_el_zb[0][3], 0., m_el_zb[1][3], 0.,     0., 0., m_el_zb[2][3], 0., m_el_zb[3][3], 0.],
                         [0., m_el_yg[0][3], 0., 0., 0., m_el_yg[1][3],     0., m_el_yg[2][3], 0., 0., 0., m_el_yg[3][3]]])


        version2_m_el = m_el
        # print("\n VERSION2 - m_el")
        # print(np.array2string(version2_m_el, precision=3, separator=',', suppress_small=True))

        print("\n NORM - between m_el_ versions:")
        print(np.linalg.norm(version1_m_el - version2_m_el))
        print()
        # wait = input("check...")

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

        # # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        # print(" If mass global is positive definite should return True")
        # print(np.all(np.linalg.eigvals(m_glob) > 0))
        # wait = input("check...")

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
        
        #
        # VERSION 1
        #
        # NOTE: checking out alternative implementation
        # according to https://mediatum.ub.tum.de/doc/1072355/file.pdf
        # description and implementation seems correct
        # NOTE: find out where the formulation for the mass comes from, stiffness seems standard

        t = 45.0 # t: beam thickness (y) [m]
        h = 30.0 # h: beam height (z) [m]
        rho = 160.0 # density of steel [kg/mˆ3]
        E = 2.861e8 # E: Young's modulus of steel [N/mˆ2]
        nu = 3/10 # nu: Poisson's ratio

        G = E/2/(1+nu) # G: Shear modulus [N/mˆ2]
        l = level_height # l: beam element length
        A = t*h # beam area [mˆ2]
        ASy = 5/6*A    
        ASz = 5/6*A # effective area of shear
        Iy = 1/12*h**3*t    
        Iz = 1/12*t**3*h # second moments of area [mˆ4]
        Ip = 1/12*t*h*(h**2+t**2) # polar moment of inertia [mˆ4]
        It = min([h,t])**3 *max([h,t])/7 # torsion constant [mˆ4]
        Py = 12*E*Iz/(G*ASy*l**2) #
        Pz = 12*E*Iy/(G*ASz*l**2) #

        K11 = np.zeros((6,6))
        K11[0][0] = E*A/l
        K11[1][1] = 12*E*Iz/(l**3*(1+Py))
        K11[2][2] = 12*E*Iy/(l**3*(1+Pz))
        K11[3][3] = G*It/l
        K11[4][4] = (4+Pz)*E*Iy/(l*(1+Pz))
        K11[5][5] = (4+Py)*E*Iz/(l*(1+Py))
        K11[1][5] = 6*E*Iz/(l**2*(1+Py))
        K11[5][1] = K11[1][5]
        K11[2][4] =-6*E*Iy/(l**2*(1+Pz))
        K11[4][2] = K11[2][4] 

        K22 = -K11 + 2*np.diag(np.diag(K11))

        K21 = K11 - 2*np.diag(np.diag(K11))
        K21[4][4] = (2-Pz)*E*Iy/(l*(1+Pz))
        K21[5][5] = (2-Py)*E*Iz/(l*(1+Py))
        K21[1][5] =-K21[5][1]
        K21[2][4] =-K21[4][2]

        k_el = np.zeros((2*6,2*6))
        # upper left
        k_el[0:6,0:6] += K11
        # lower left
        k_el[6:12,0:6] += K21
        # upper right
        k_el[0:6,6:12] += np.transpose(K21)
        # lower right
        k_el[6:12,6:12] += K22

        version1_k_el = k_el
        # print("\n VERSION1 - k_el")
        # print(np.array2string(version1_k_el, precision=3, separator=',', suppress_small=True))


        #
        # VERSION 2
        #
        # NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
        # seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
        # implemented mass matrices similar to the stiffness one
        
        # stifness values for one level
        # define component-wise to have enable better control for various optimization parameters

        length = l
        # axial stiffness - along axis x - here marked as x
        k_x = E*A/l
        k_x_11 = 1.0
        k_x_12 = -1.0
        k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                 [k_x_12, k_x_11]])
        # torsion stiffness - around axis x - here marked as alpha - a
        k_a = G*It/l  # G*K/l
        k_a_11 = 1.0
        k_a_12 = -1.0
        k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                 [k_a_12, k_a_11]])
        # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
        beta_yg = Py
        k_yg = E*Iz/(1+beta_yg)/l**3
        #
        k_yg_11 = 12.
        k_yg_12 = 6*length
        k_yg_13 = -12.
        k_yg_14 = 6.*length
        #
        k_yg_22 = (4.+beta_yg) * length **2
        k_yg_23 = -6. * length
        k_yg_24 = (2-beta_yg) * length ** 2
        #
        k_yg_33 = 12.
        k_yg_34 = -6. * length
        #
        k_yg_44 = (4+beta_yg) * length ** 2
        #
        k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
                                   [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
                                   [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
                                   [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

        # bending - displacement along axis z, rotations around axis y - here marked as beta - b
        beta_zb = Pz
        k_zb = E*Iy/(1+beta_zb)/l**3
        #
        k_zb_11 = 12.
        k_zb_12 = -6*length
        k_zb_13 = -12.
        k_zb_14 = -6.*length
        #
        k_zb_22 = (4.+beta_zb) * length **2
        k_zb_23 = 6. * length
        k_zb_24 = (2-beta_zb) * length ** 2
        #
        k_zb_33 = 12.
        k_zb_34 = 6. * length
        #
        k_zb_44 = (4+beta_zb) * length ** 2
        #
        k_el_zb = k_zb * np.array([[k_zb_11, k_zb_12, k_zb_13, k_zb_14],
                                   [k_zb_12, k_zb_22, k_zb_23, k_zb_24],
                                   [k_zb_13, k_zb_23, k_zb_33, k_zb_34],
                                   [k_zb_14, k_zb_24, k_zb_34, k_zb_44]])

        # assemble all components
        k_el = np.array([[k_el_x[0][0], 0., 0., 0., 0., 0.,                 k_el_x[0][1], 0., 0., 0., 0., 0.],
                         [0., k_el_yg[0][0], 0., 0., 0., k_el_yg[0][1],     0., k_el_yg[0][2], 0., 0., 0., k_el_yg[0][3]],
                         [0., 0., k_el_zb[0][0], 0., k_el_zb[0][1], 0.,     0., 0., k_el_zb[0][2], 0., k_el_zb[0][3], 0.],
                         [0., 0., 0., k_el_a[0][0], 0., 0.,                 0., 0., 0., k_el_a[0][1], 0., 0.],
                         [0., 0., k_el_zb[0][1], 0., k_el_zb[1][1], 0.,     0., 0., k_el_zb[1][2], 0., k_el_zb[1][3], 0.],
                         [0., k_el_yg[0][1], 0., 0., 0., k_el_yg[1][1],     0., k_el_yg[1][2], 0., 0., 0., k_el_yg[1][3]],
                         
                         [k_el_x[1][0], 0., 0., 0., 0., 0.,                 k_el_x[1][1], 0., 0., 0., 0., 0.],
                         [0., k_el_yg[0][2], 0., 0., 0., k_el_yg[1][2],     0., k_el_yg[2][2], 0., 0., 0., k_el_yg[2][3]],
                         [0., 0., k_el_zb[0][2], 0., k_el_zb[1][2], 0.,     0., 0., k_el_zb[2][2], 0., k_el_zb[2][3], 0.],
                         [0., 0., 0., k_el_a[1][0], 0., 0.,                 0., 0., 0., k_el_a[1][1], 0., 0.],
                         [0., 0., k_el_zb[0][3], 0., k_el_zb[1][3], 0.,     0., 0., k_el_zb[2][3], 0., k_el_zb[3][3], 0.],
                         [0., k_el_yg[0][3], 0., 0., 0., k_el_yg[1][3],     0., k_el_yg[2][3], 0., 0., 0., k_el_yg[3][3]]])

        version2_k_el = k_el
        # print("\n VERSION2 - k_el")
        # print(np.array2string(version2_k_el, precision=3, separator=',', suppress_small=True))

        print("\n NORM - k_el versions:")
        print(np.linalg.norm(version1_k_el - version2_k_el))
        print()
        # wait = input("check...")

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