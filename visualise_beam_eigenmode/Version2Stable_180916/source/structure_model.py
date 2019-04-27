#===============================================================================
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
#===============================================================================

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from functools import partial

class Simplified2DCantileverStructure(object):
    def __init__(self, m, b, k, nodal_coordinates, name, category):
        """
        Sets up the main structural properties (matrices) 
        and includes the nodal coordinates (in height) 
        as well as tags (strings) for naming and category.

        m, b, k matrices -> setup with the Dirichlet BC included
        
        nodal_coordinates dictionary with keys "x0", "y0" undformed
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
        # a chosen category - one of "SDoF", "MDoFShear", "MDoFBeam"
        if category in ['SDoF','MDoFShear','MDoFBeam', 'MDoFMixed']:
            self.category = category
        else:
            err_msg =  'The requested category \"' + category + '\" is not supported in Simplified2DCantileverStructure\n'
            err_msg += 'Available options are: \"SDoF\",\"MDoFShear\", \"MDoFBeam\", \"MDoFMixed\"'
            raise Exception(err_msg)

class SDoFModel(Simplified2DCantileverStructure):
    """
    A single-degree-of-freedom SDoF model        
    """
    def __init__(self, m=1., target_freq=1., zeta=0.05, level_height=3.5,
                 num_of_levels=1, name="DefaultSDoFModel"):

        k = self._calculate_stiffness(m,target_freq)
        b = self._calculate_damping(m,k, zeta)

        height_coordinates = self._get_nodal_coordinates(level_height, num_of_levels) 

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(np.array([[m]]), 
                         np.array([[b]]), 
                         np.array([[k]]), 
                         nodal_coordinates, 
                         name, 
                         category='SDoF')

    def _calculate_stiffness(self, m, target_freq):
        """
        Calculate stiffness k
        """
        print("Calculating stiffness k in SDoFModel derived class \n") 
        return m * (target_freq * 2 * np.pi)**2

    def _calculate_damping(self, m, k, zeta):
        """
        Calculate damping b
        """
        print("Calculating damping b in SDoFModel derived class \n")
        return zeta * 2.0 * np.sqrt(m * k)

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1,num_of_levels+1)
        return nodal_coordinates

    
class MDoFShearModel(Simplified2DCantileverStructure):
    
    """
    A multi-degree-of-freedom MDoF model assuming
    shear-type deformations using an extension of the
    spring-mass system

    ATTENTION:
    For this model a homogenous distribution of mass,
    stiffness and damping is a premise. For other cases
    this model is not adequate and changes need to be done.
    """

    def __init__(self, 
                 rho=5.,
                 area=10.,
                 target_freq=1.,
                 target_mode=1,
                 zeta=0.05,
                 level_height=3.5,
                 num_of_levels=3,
                 name="DefaultMDoFShearModel"):

        m = self._calculate_mass(rho, area, level_height, num_of_levels)
        k = self._calculate_stiffness(m, level_height, num_of_levels, target_freq, target_mode)
        b = self._calculate_damping(m, k, zeta)

        height_coordinates = self._get_nodal_coordinates(level_height, num_of_levels) 

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoFShear')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1,num_of_levels+1)
        return nodal_coordinates


    def _calculate_mass(self, rho, area, level_height, num_of_levels):
        """
        Getting the consistant mass matrix 
        """
        # mass values for one level
        length = level_height
        m_const = rho * area * length / 2 
        m_elem = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

        # global mass matrix initialization with zeros
        m_glob = np.zeros((num_of_levels + 1, num_of_levels + 1))
        # fill global mass matrix entries
        for i in range(num_of_levels):
            m_temp = np.zeros((num_of_levels +1, num_of_levels + 1))
            m_temp[i:i + 2, i:i + 2] = m_elem
            m_glob += m_const * m_temp

        # remove the fixed degrees of freedom -> applying Dirichlet BC implicitly
        for i in [0, 1]:
            m_glob = np.delete(m_glob, 0, axis=i)

        # return mass matrix
        return m_glob

    def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
        """
        Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all 
        the elements and the value is calculated using an optimization (or "tuning")
        for a target frequency of a target mode.
        """
        print("Calculating stiffness k in MDoFShearModel derived class \n")

        # setup k_scalar_guess as the input for the standard k for a shear-type
        # MDoF
        k_scalar_guess = 1.

        # using partial to fix some parameters for the
        # self._calculate_frequency_for_current_scalar_k()
        optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar, 
                                       m, 
                                       level_height, 
                                       num_of_levels, 
                                       target_freq, 
                                       target_mode)

        print("Optimization for the target k matrix in MDoFShearModel \n")
        minimization_result = minimize(optimizable_function,
                                       k_scalar_guess,
                                       options={'disp': True})
        print("")

        k_scalar_opt = minimization_result.x[0]
     
        return self._assemble_k(level_height, num_of_levels, k_scalar_opt)

    def _assemble_k(self, level_height, num_of_levels, k_scalar):
        """
        For the MDoFBeam model stiffness distribution according to beam theory is assumed
        the stiffness matrix is asembled with the k_scalar calculated.
        """

        k_const = k_scalar 
        # stifness values for one level
        k_elem = np.array([[1.0, -1.0],
                           [-1.0, 1.0]])

        # global stiffness matrix initialization with zeros
        k_glob = np.zeros((num_of_levels + 1, num_of_levels + 1))
        # fill global stiffness matix entries
        for i in range(num_of_levels):
            k_temp = np.zeros(
                (num_of_levels +1, num_of_levels + 1))
            k_temp[i:i + 2, i:i + 2] = k_elem
            k_glob += k_const * k_temp

        # remove the fixed degrees of freedom -> applying Dirichlet BC implicitly
        for i in [0, 1]:
            k_glob = np.delete(k_glob, 0, axis=i)

        # return stiffness matrix
        return k_glob


    def _calculate_damping(self, m, k, zeta):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """
        print("Calculating damping b in MDoFShearModel derived class \n")
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

    def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
        k = self._assemble_k(level_height, num_of_levels, k_scalar)

        # TODO: try to avoid this code duplication
        # raw results
        eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        
        current_target_freq = eig_freqs[eig_freqs_sorted_indices[target_mode-1]]

        return np.sqrt((target_freq - current_target_freq) **2) / target_freq


class MDoFBeamModel(Simplified2DCantileverStructure):    
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
                 name="DefaultMDofBeamModel"):
       

        m = self._calculate_mass(rho, area, level_height, num_of_levels)
        k = self._calculate_stiffness(m, level_height, num_of_levels, target_freq, target_mode)
        b = self._calculate_damping(m, k, zeta)

        height_coordinates = self._get_nodal_coordinates(level_height, num_of_levels) 

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoFBeam')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1,num_of_levels+1)
        return nodal_coordinates
    
    def _calculate_mass(self, rho, area, level_height, num_of_levels):
        """
        Getting the consistant mass matrix based on analytical integration
        """
        # mass values for one level
        length = level_height
        m_const = rho * area * length / 420
        m_elem = np.array([[         156,     22 * length,           54,    -13 * length],
                           [ 22 * length,  4 * length **2,  13 * length, -3 * length **2],
                           [          54,     13 * length,          156,    -22 * length],
                           [-13 * length, -3 * length **2, -22 * length,  4 * length **2]])

        # global mass matrix initialization with zeros
        m_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))
        # fill global mass matrix entries
        for i in range(num_of_levels):
            m_temp = np.zeros(
                (2 * num_of_levels + 2, 2 * num_of_levels + 2))
            m_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = m_elem
            m_glob += m_const * m_temp

        # remove the fixed degrees of freedom
        for dof in [1, 0]:
            for i in [0, 1]:
                m_glob = np.delete(m_glob, dof, axis=i)

        # return mass matrix
        return m_glob

    def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
        """
        Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all 
        the elements and the value is calculated using an optimization (or "tuning")
        for a target frequency of a target mode.
        """
        print("Calculating stiffness k in MDoFBeamModel derived class \n")

        # setup k_scalar_guess as the input for the standard k for a shear-type
        # MDoF
        k_scalar_guess = 1000.

        # using partial to fix some parameters for the
        # self._calculate_frequency_for_current_scalar_k()
        optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar, 
                                       m, 
                                       level_height, 
                                       num_of_levels, 
                                       target_freq, 
                                       target_mode)

        print("Optimization for the target k matrix in MDoFBeamModel \n")
        minimization_result = minimize(optimizable_function,
                                       k_scalar_guess, method='Powell',
                                       options={'disp': True})
        
        # returning only one value!
        k_scalar_opt = minimization_result.x

        # retutning minimization value to class, make it callable
        self.k_scalar = k_scalar_opt
    
        return self._assemble_k(level_height, num_of_levels, k_scalar_opt)

    def _assemble_k(self, level_height, num_of_levels, k_scalar):
        """
        For the MDoFBeam model stiffness distribution according to beam theory is assumed
        the stiffness matrix is asembled with the k_scalar calculated.
        """
        # k_scalar = EI
        length = level_height
        k_const = k_scalar / pow(length, 3)
        # stifness values for one level
        k_elem = np.array([[        12,     6 * length,         -12,     6 * length],
                           [6 * length, 4 * length **2, -6 * length, 2 * length **2],
                           [       -12,    -6 * length,          12,    -6 * length],
                           [6 * length, 2 * length **2, -6 * length, 4 * length **2]])

        # global stiffness matrix initialization with zeros
        k_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))
        # fill global stiffness matix entries
        for i in range(num_of_levels):
            k_temp = np.zeros(
                (2 * num_of_levels + 2, 2 * num_of_levels + 2))
            k_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = k_elem
            k_glob += k_const * k_temp

        # remove the fixed degrees of freedom
        for dof in [1, 0]:
            for i in [0, 1]:
                k_glob = np.delete(k_glob, dof, axis=i)

        # return stiffness matrix
        return k_glob

    def _calculate_damping(self, m, k, zeta):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """
        print("Calculating damping b in MDoFBeamModel derived class \n")
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

    def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
        k = self._assemble_k(level_height, num_of_levels, k_scalar)

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

        return (target_freq - current_target_freq) **2 / target_freq**2

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


class MDoFMixedModel(Simplified2DCantileverStructure):    
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
                 k_beam = 1,
                 name="DefaultMDofMixedModel"):
       

        m = self._calculate_mass(rho, area, level_height, num_of_levels)
        k = self._calculate_stiffness(m, level_height, num_of_levels, target_freq, target_mode, k_beam)
        b = self._calculate_damping(m, k, zeta)

        height_coordinates = self._get_nodal_coordinates(level_height, num_of_levels) 

        nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
                             "y0": height_coordinates,
                             "x": None,
                             "y": None}

        super().__init__(m, b, k, nodal_coordinates, name, category='MDoFMixed')

    def _get_nodal_coordinates(self, level_height, num_of_levels):
        nodal_coordinates = level_height * np.arange(1,num_of_levels+1)
        return nodal_coordinates
    
    def _calculate_mass(self, rho, area, level_height, num_of_levels):
        """
        Getting the consistant mass matrix based on analytical integration
        """
        # mass values for one level
        length = level_height
        alpha = 1/100 # generally between 0 and 1/100
        m_const = rho * area * length
        m_elem = np.array([[ 1/2, 0,                 0,   0],
                           [ 0,   alpha * length**2, 0,   0],
                           [ 0,   0,                 1/2, 0],
                           [ 0,   0,                 0,   alpha * length **2]])

        # global mass matrix initialization with zeros
        m_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))
        # fill global mass matrix entries
        for i in range(num_of_levels):
            m_temp = np.zeros(
                (2 * num_of_levels + 2, 2 * num_of_levels + 2))
            m_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = m_elem
            m_glob += m_const * m_temp

        # remove the fixed degrees of freedom
        for dof in [1, 0]:
            for i in [0, 1]:
                m_glob = np.delete(m_glob, dof, axis=i)

        # return mass matrix
        return m_glob

    def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode, k_beam):
        """
        Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all 
        the elements and the value is calculated using an optimization (or "tuning")
        for a target frequency of a target mode.
        """
        print("Calculating stiffness k in MDoFMixedModel derived class \n")

        # setup k_scalar_guess as the input for the standard k for a shear-type
        # MDoF
        k_scalar_guess = 1.

        # using partial to fix some parameters for the
        # self._calculate_frequency_for_current_scalar_k()
        optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar, 
                                       m, 
                                       level_height, 
                                       num_of_levels, 
                                       target_freq, 
                                       target_mode,
                                       k_beam)

        print("Optimization for the target k matrix in MDoFMixedModel \n")
        minimization_result = minimize(optimizable_function,
                                       k_scalar_guess, method='Powell',
                                       options={'disp': True})
        
        # returning only one value!
        k_scalar_opt = minimization_result.x
        print("K spring optimized: ", k_scalar_opt)
        print("K beam: ", k_beam)
        input()
        return self._assemble_k(level_height, num_of_levels, k_scalar_opt, k_beam)

    def _assemble_k(self, level_height, num_of_levels, k_scalar, k_beam):
        """
        For the MDoFMixed model stiffness distribution according to beam theory is assumed
        the stiffness matrix is asembled with the k_scalar calculated.
        """

        # element length / level height
        length = level_height

        # calculated beam stiffness
        k_const_beam = k_beam/ pow(length, 3)

        # searched spring stiffness
        k_const_spring = k_scalar
       
        # stifness values for one level
        k_beam_elem = np.array([[        12,     6 * length,         -12,     6 * length],
                           [6 * length, 4 * length **2, -6 * length, 2 * length **2],
                           [       -12,    -6 * length,          12,    -6 * length],
                           [6 * length, 2 * length **2, -6 * length, 4 * length **2]])

        k_spring_elem = np.array([[1, 0, -1, 0],
                                  [0, 0 , 0, 0],
                                  [-1, 0, 1, 0],
                                  [0, 0, 0, 0]])

        # global stiffness matrix initialization with zeros
        k_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))

        # fill global stiffness matix entries
        for i in range(num_of_levels):
            k_temp = np.zeros(
                (2 * num_of_levels + 2, 2 * num_of_levels + 2))
            k_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = k_const_beam * k_beam_elem + k_const_spring * k_spring_elem
            k_glob += k_temp

        # remove the fixed degrees of freedom
        for dof in [1, 0]:
            for i in [0, 1]:
                k_glob = np.delete(k_glob, dof, axis=i)

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

    def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar, k_beam):
        k = self._assemble_k(level_height, num_of_levels, k_scalar, k_beam)

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

        return (target_freq - current_target_freq) **2 / target_freq**2

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
