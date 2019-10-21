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
Last update: 09.07.2018
'''
# ===============================================================================

import numpy as np
from scipy import linalg
# TODO only use minimize, make dependency on minimize_scalar work with that instead
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import matplotlib.pyplot as plt

from source.model.structure_model import StraightBeam
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
from source.auxiliary.global_definetions import *


CUST_MAGNITUDE = 2


class OptimizableStraightBeam(object):
    """
    A 2D/3D prismatic homogenous isotropic Timoshenko beam element
    Including shear and rotationary inertia and deformation
    Using a consistent mass formulation

    Definition of axes:
        1. Longitudinal axis: x with rotation alpha around x
        2. Transversal axes: 
            y with rotation beta around y 
            z with rotation gamma around z

    Degrees of freedom DoFs
        1. 2D: displacements x, y, rotation g(amma) around z 
            -> element DoFs for nodes i and j of one element
                [0, 1, 2, 3, 4, 5, 6] = [x_i, y_i, g_i, 
                                        x_j, y_j, g_j]

        2. 3D: displacements x, y, z, rotationS a(lpha), b(eta), g(amma) 
            -> element DoFs for nodes i and j of one element
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = [x_i, y_i, z_i, a_i, b_i, g_i,
                                                        x_j, y_j, z_j, a_j, b_j, g_j]

    TODO:
        1. add a parametrization to include artificial (and possibly also local) 
            incrementation of stiffness and mass (towards a shear beam and/or point mass/stiffness)
        2. add a parametrization for tunig to eigen frequencies and mode shapes
        3. add a parametrization to be able to specify zones with altering mass ditribution
        4. add a parametrization to be able to handle a certain distribution or area or similar along length
        5. extend to be able to handle non-centric centre of elasticity with respect to center of mass
        6. test unti how many elements it works, also specific cases with analystical solutions
    """

    OPT_FCTR = 100

    THRESHOLD = 1e-8

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        # TODO: will assign this mass if no value provided, figure out bettwe way
        "density_for_total_mass": 0.0,
        "youngs_modulus_for": {},
        "geometric_properties_for": {}
                }

    def __init__(self, model, parameters):

        if not(isinstance(model, StraightBeam)):
            err_msg = "The proivded model is of type \"" + \
                str(type(model)) + "\"\n"
            err_msg += "Has to be of type \"<class \'StraigthBeam\'>\""
            raise Exception(err_msg)
        self.model = model

        validate_and_assign_defaults(OptimizableStraightBeam.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        print('BEFORE OPTIMIZATION')
        self.model.identify_decoupled_eigenmodes(print_to_console=True)
        print()

        print('Found need for adapting structure for target values')

        # if a target mass is set, the density will be adjusted, no additional dependencies
        if 'density_for_total_mass' in self.parameters:
            print('DENSITY OPTIMIZATION')

            target_total_mass = self.parameters["density_for_total_mass"]
            print('Adapting density for target total mass: ', target_total_mass)

            self.adjust_density_for_target_total_mass(target_total_mass)

        # if generically a target mode and frequency is set, the e-modul will be adjusted, g-modul recomputed
        if 'youngs_modulus_for' in self.parameters and self.parameters['youngs_modulus_for']:
            print('YOUNG\'S MODULUS OPTIMIZATION')

            target_mode = self.parameters["youngs_modulus_for"]["eigenmode"]
            target_freq = self.parameters["youngs_modulus_for"]["eigenfrequency"]
            print('Adapting young\'s modulus for target eigenfrequency: ' +
                  str(target_freq) + ' and mode: ' + str(target_mode))

            self.adjust_e_modul_for_target_eigenfreq(
                target_freq, target_mode, True)

        if 'geometric_properties_for' in self.parameters and self.parameters['geometric_properties_for']:

            modes_to_consider = self.parameters["geometric_properties_for"]["consider_decomposed_modes"]
            modes_possible_to_consider = [
                *MODE_CATEGORIZATION[self.model.domain_size].keys()]
            diff_list = np.setdiff1d(
                modes_to_consider, modes_possible_to_consider)
            if len(diff_list) != 0:
                err_msg = "The element(s) \"" + ', '.join(diff_list) + "\"\n"
                err_msg += "in provided modes to consider \"" + \
                    ', '.join(modes_to_consider) + "\n"
                err_msg += "\" are not available for consideration\n"
                err_msg += "Choose one or more of: \""
                err_msg += ', '.join(modes_possible_to_consider) + "\"\n"
                raise Exception(err_msg)

            # IMPORTANT: a adaptation/tuning/optimization has to be done in the follopwing order
            # TODO: check dependencies for order of execution and necessery updates

            # 1. LONGITUDINAL
            # TODO: optimize for area -> update rho to match total mass, also update a_sy, a_sz

            # NOTE: it seems to need total mass and in general difficult/insesitive to tuning...
            if 'longitudinal' in self.parameters["geometric_properties_for"]["consider_decomposed_modes"]:
                print('LONGITUDINAL OPTIMIZATION')

                identifier = 'longitudinal'

                id_idx = self.parameters["geometric_properties_for"]["consider_decomposed_modes"].index(
                    identifier)
                target_mode = self.parameters["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                target_freq = self.parameters["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]

                self.adjust_longitudinal_stiffness_for_target_eigenfreq(
                    target_freq, target_mode, True)

            # 2. and 3. - on of SWAY_Y and/or SWAY_Z
            # TODO: optimize for iz, iy (maybe also extend to a_sy, a_sz -> multi design param opt) -> update ip with new values, also pz, py
            if 'sway_y' in self.parameters["geometric_properties_for"]["consider_decomposed_modes"]:
                print('SWAY_Y OPTIMIZATION')

                identifier = 'sway_y'

                id_idx = self.parameters["geometric_properties_for"]["consider_decomposed_modes"].index(
                    identifier)
                target_mode = self.parameters["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                target_freq = self.parameters["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]
                # try:
                #     bending_shear_identifier = self.parameters["geometric_properties_for"]["partition_shear_bending"][0]
                # except:
                #     bending_shear_identifier = 0
                # This is not being used currently. 

                self.adjust_sway_y_stiffness_for_target_eigenfreq(
                    target_freq, target_mode, True)

            if 'sway_z' in self.parameters["geometric_properties_for"]["consider_decomposed_modes"]:
                print('SWAY_Z OPTIMIZATION')

                identifier = 'sway_z'

                id_idx = self.parameters["geometric_properties_for"]["consider_decomposed_modes"].index(
                    identifier)
                target_mode = self.parameters["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                target_freq = self.parameters["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]

                self.adjust_sway_z_stiffness_for_target_eigenfreq(
                    target_freq, target_mode, True)

            # 4. TORSIONAL
            # TODO: optimize for it -> needs updated model from previous cases
            if 'torsional' in self.parameters["geometric_properties_for"]["consider_decomposed_modes"]:
                print('TORSIONAL OPTIMIZATION')

                identifier = 'torsional'

                id_idx = self.parameters["geometric_properties_for"]["consider_decomposed_modes"].index(
                    identifier)
                target_mode = self.parameters["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                target_freq = self.parameters["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]

                self.adjust_torsional_stiffness_for_target_eigenfreq(
                    target_freq, target_mode, True)

        print('AFTER OPTIMIZATION')
        self.model.identify_decoupled_eigenmodes(print_to_console=True)
        print()

    def generic_material_stiffness_objective_function(self, target_freq, target_mode, initial_e, multiplier_fctr):

        for e in self.model.elements:
            e.E = multiplier_fctr * initial_e

            # NOTE: do not forget to update G and further dependencies
            e.G = e.E / 2 / (1 + e.nu)
            e.evaluate_relative_importance_of_shear()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[target_mode-1]] - target_freq)**2 / target_freq**2

    def adjust_density_for_target_total_mass(self, target_total_mass):

        # calculate to be sure to have most current
        self.model.calculate_total_mass(True)

        corr_fctr = target_total_mass / self.model.parameters['m_tot']

        self.model.parameters['rho'] *= corr_fctr

        # re-calculate and print to console
        print('AFTER TUNED DENSITY')
        self.model.calculate_total_mass(True)

    def adjust_e_modul_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_e = self.model.parameters['e']

        # using partial to fix some parameters for the
        optimizable_function = partial(self.generic_material_stiffness_objective_function,
                                       target_freq,
                                       target_mode,
                                       initial_e)

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR))

        # returning only one value!
        opt_e_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL e:', initial_e)
            print('OPTIMIZED e: ', opt_e_fctr * initial_e)
            print()

    def adjust_longitudinal_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_a = (e.A for e in self.model.elements)
        # assuming a linear dependency of shear areas
        initial_a_sy = (e.Asy for e in self.model.elements)
        initial_a_sz = (e.Asz for e in self.model.elements)

        # using partial to fix some parameters for the
        optimizable_function = partial(self.longitudinal_geometric_stiffness_objective_function,
                                       target_freq,
                                       target_mode,
                                       initial_a,
                                       initial_a_sy,
                                       initial_a_sz)

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR))

        # returning only one value!
        opt_a_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL a:', ', '.join([str(val) for val in initial_a]))
            print()
            print('OPTIMIZED a: ', ', '.join(
                [str(opt_a_fctr * val) for val in initial_a]))
            print()
            print('FACTOR: ', opt_a_fctr)
            print()

    def longitudinal_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_a, initial_a_sy, initial_a_sz, multiplier_fctr):

        for e in self.model.elements:
            e.A = multiplier_fctr * initial_a[e.index]
            # assuming a linear dependency of shear areas
            e.Asy = multiplier_fctr * initial_a_sy[e.index]
            e.Asz = multiplier_fctr * initial_a_sz[e.index]

            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()

        # NOTE: it seems to need total mass and in general difficult/insesitive to tuning...
        # TODO:
        # self.adjust_density_for_target_total_mass(target_total_mass)

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'longitudinal'
        mode_ids = self.model.mode_identification_results[identifier]

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[mode_ids[0]-1]] - target_freq)**2 / target_freq**2

    def adjust_sway_y_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_iy = list(e.Iy for e in self.model.elements)
        initial_a_sz = list(e.Asz for e in self.model.elements)

        # using partial to fix some parameters for the
        optimizable_function = partial(self.bending_y_geometric_stiffness_objective_function,
                                       target_freq,
                                       target_mode,
                                       initial_iy, 
                                       initial_a_sz)
        init_guess = (1.0, 1.0)

        bnds_iy = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)#(1/8,8)
        bnds_a_sz = (0.4,1.0)#(1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)#(1/15,15)

        minimization_result = minimize(optimizable_function,
                                              init_guess,
                                              method='L-BFGS-B',#'SLSQP',#
                                              bounds=(bnds_iy,bnds_a_sz))

        # returning only one value!
        opt_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL iy:', ', '.join([str(val) for val in initial_iy]))
            print()
            print('OPTIMIZED iy: ', ', '.join(
                [str(opt_fctr[0] * val) for val in initial_iy]))
            print('INITIAL a_sz:', ', '.join([str(val) for val in initial_a_sz]))
            print()
            print('OPTIMIZED a_sz: ', ', '.join(
                [str(opt_fctr[1] * val) for val in initial_a_sz]))
            print()
            print('FACTORS: ', ', '.join([str(val) for val in opt_fctr]))
            print()
            

    def bending_y_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iy, initial_a_sz, multiplier_fctr):

        for e in self.model.elements:
            e.Iy = multiplier_fctr * initial_iy[e.index]
            # assuming a linear dependency of shear areas
            e.Asz = multiplier_fctr * initial_a_sz[e.index]
            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'sway_y'
        mode_ids = self.model.mode_identification_results[identifier]

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[mode_ids[0]-1]] - target_freq)**2 / target_freq**2

    def adjust_sway_z_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):

        initial_iz = list(e.Iz for e in self.model.elements)
        initial_a_sy = list(e.Asy for e in self.model.elements)

        # using partial to fix some parameters for the
        optimizable_function = partial(self.bending_z_geometric_stiffness_objective_function,
                                       target_freq,
                                       target_mode,
                                       initial_iz, 
                                       initial_a_sy)
        initi_guess = (1.0, 1.0)

        bnds_iz = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)#(1/8,8)
        bnds_a_sy = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)#(1/15,15)



        minimization_result = minimize(optimizable_function,
                                              initi_guess,
                                              method='L-BFGS-B',
                                              bounds=(bnds_iz,bnds_a_sy))

        # returning only one value!
        opt_iz_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL iz:', ', '.join([str(val) for val in initial_iz]))
            print()
            print('OPTIMIZED iz: ', ', '.join(
                [str(opt_iz_fctr[0] * val) for val in initial_iz]))
            print()
            print('INITIAL a_sy:', ', '.join([str(val) for val in initial_a_sy]))
            print()
            print('OPTIMIZED a_sy: ', ', '.join(
                [str(opt_iz_fctr[1] * val) for val in initial_a_sy]))
            print('FACTOR: ', opt_iz_fctr)
            print()

    def bending_z_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iz, initial_a_sy, multiplier_fctr):

        for e in self.model.elements:
            e.Iz = multiplier_fctr * initial_iz[e.index]
            # assuming a linear dependency of shear areas
            e.Asy = multiplier_fctr * initial_a_sy[e.index]

            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'sway_z'
        mode_ids = self.model.mode_identification_results[identifier]

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[mode_ids[0]-1]] - target_freq)**2 / target_freq**2

    def adjust_torsional_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_it = list(e.It for e in self.model.elements)
        initial_ip = list(e.Ip for e in self.model.elements)

        # NOTE: single parameter optimization seems not to be enough

        # using partial to fix some parameters for the
        optimizable_function = partial(self.torsional_geometric_stiffness_objective_function,
                                       target_freq,
                                       target_mode,
                                       initial_it,
                                       initial_ip)

        # NOTE: some additional reduction factor so that ip gets changes less

        init_guess = (1.0, 1.0)

        # NOTE: this seems not to be enough
        # bnds_it = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)
        # bnds_ip = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)

        # NOTE: seems that the stiffness contribution takes lower bound, the inertia one the upper bound
        bnds_it = (1/7, 7)
        bnds_ip = (1/11, 11)

        # NOTE: TNC, SLSQP, L-BFGS-B seems to work with bounds correctly, COBYLA not
        minimization_result = minimize(optimizable_function,
                                       init_guess,
                                       method='L-BFGS-B',
                                       bounds=(bnds_it, bnds_ip))

        # returning only one value!
        opt_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL it:', ', '.join([str(val) for val in initial_it]))
            print('OPTIMIZED it: ', ', '.join(
                [str(opt_fctr[0] * val) for val in initial_it]))
            print()
            print('INITIAL ip:', ', '.join([str(val) for val in initial_ip]))
            print('OPTIMIZED ip: ', ', '.join(
                [str(opt_fctr[1] * val) for val in initial_ip]))
            print()
            print('FACTORS: ', ', '.join([str(val) for val in opt_fctr]))
            print()

    def torsional_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_it, initial_ip, multiplier_fctr):

        for e in self.model.elements:
            e.It = initial_it[e.index]
            e.Ip = initial_ip[e.index]

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'torsional'
        mode_ids = self.model.mode_identification_results[identifier]

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[mode_ids[0]-1]] - target_freq)**2 / target_freq**2
