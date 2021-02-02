import numpy as np
from scipy import linalg
# TODO only use minimize, make dependency on minimize_scalar work with that instead
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import matplotlib.pyplot as plt
from os.path import join as os_join
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
#import numpy.polynomial.polynomial.Polynomial.fit as polyfit

from source.model.structure_model import StraightBeam
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
import source.auxiliary.global_definitions as GD
from source.analysis.static_analysis import StaticAnalysis
from source.auxiliary.auxiliary_functionalities import stop_run
import source.postprocess.plotter_utilities as plotter_utilities
from source.auxiliary.other_utilities import get_signed_maximum
import source.auxiliary.CAARC_utilities as caarc

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

        validate_and_assign_defaults(
            OptimizableStraightBeam.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        # storing the CAARC eigenmodes in the model object to better acces them everywhere
        self.model.CAARC_eigenmodes = caarc.get_CAARC_eigenmodes()

# # FLAGS FOR OWN OPTIMIZATION

        # Faktorisierung der Caarc eigenmodes um masse normalisierte werte zu erhalten 
        self.CAARC_eigenform_factorization = True

        # NOTE: anzahl der element des akutellen models ist hier wichtig 
        self.adjust_across_stiffness = False
        self.adjust_density_elem_wise = False

        # NOTE: geht gerade nur für ein model mit 4 elementen
        self.optimize_node_dof_wise_for_CAARC_eigenform = False

        # optimierungen mit statischen Verschiebungen als target
        self.stiffness_for_static_displacement = False

        self.coupling_entries_for_rotation = False

# # CUSTOM OBJECTS OF EIGNEVALUE_ANALYSIS AND STATIC_ANALYSIS 
        self.static_analysis_params_custom= {
                "type" : "static_analysis",
                "settings": {},
                "input":{
                    "help":"provide load file in the required format - either some symbolic generated or time step from dynamic",
                    "file_path": "input/force/generic_building/static_force_4_nodes_at_4_in_y.npy",
                    "is_time_history_file" : False,
                    "selected_time_step" : 15000
                },
                "output":{
                    "plot": ["deformation", "forces"],
                    "write": ["deformation"]
                }
            }

        self.eigenvalue_analysis_params_custom = {
                "type": "eigenvalue_analysis",
                "settings": {
                    "normalization": "mass_normalized"},
                "input":{},
                "output":{
                    "eigenmode_summary": {
                        "write" : False,
                        "plot" : False},
                    "eigenmode_identification": {
                        "write" : False,
                        "plot" : False},
                    "selected_eigenmode": {
                        "plot_mode": [1,2,3],
                        "write_mode": [1,2,3],
                        "animate_mode": [1],
                        "animate_skin_model": [1]},
                    "selected_eigenmode_range": {
                        "help": "maximum 4 modes per range at a time",
                        "considered_ranges": [[1,2]],
                        "plot_range": [False, False],
                        "write_range": [True, False]}
                    }
            }

# # EXISTING OPTIMIZATION CALLS        
        print('BEFORE OPTIMIZATION')
        self.model.identify_decoupled_eigenmodes(print_to_console=True)
        print()

        print('Found need for adapting structure for target values')

        # if a target mass is set, the density will be adjusted, no additional dependencies
        if 'density_for_total_mass' in self.parameters:
            print('DENSITY OPTIMIZATION')

            target_total_mass = self.parameters["density_for_total_mass"]
            print('Adapting density for target total mass: ', target_total_mass)

            self.adjust_density_for_target_total_mass(target_total_mass, print_to_console=True)

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
                *GD.MODE_CATEGORIZATION[self.model.domain_size].keys()]
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

            # IMPORTANT: an adaptation/tuning/optimization has to be done in the follopwing order
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
                
                #self.adjust_coupling_parameter_YT_target_freq_sway_y(target_freq, target_mode)

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

# # OWN OPTIMIZATION CALLS

        if self.CAARC_eigenform_factorization:
            print ('CAARC Eigenform scaling to get mass normalized Eigenvectors\n')

            # plotter_utilities.plot_CAARC_eigenmodes(self.model.CAARC_eigenmodes,
            #                                          display_plot = True, 
            #                                          max_normed= False, 
            #                                          suptitle= 'Before Optimization')

            self.adjust_CAARC_eigenform_by_factor(plot_opt_func=False)

            print ('AFTER Factorization\n')

            for mode_id in [1,2]:
                caarc.get_m_eff(self.model.CAARC_eigenmodes, mode_id, main_direction_only = False, print_to_console= True)

            # plotter_utilities.plot_CAARC_eigenmodes(self.model.CAARC_eigenmodes,
            #                                         display_plot = True, 
            #                                         max_normed= False, 
            #                                         suptitle= 'After Optimization')

        # eigenmodes as target
        if self.adjust_across_stiffness:
            self.adjust_across_stiffness_lower_section(2)

        if self.adjust_density_elem_wise:
            self.adjust_element_density(5)

        if self.optimize_node_dof_wise_for_CAARC_eigenform:
            print ('ADJUSTING STIFFNESS FOR CAARC \n')

            self.adjust_dof_wise_stiffness_for_CAARC_eigenform(plot_opt_func = False) 
        
            print ('Finished Eigenmode Optimization of CAARC\n')

        # static displacements as target
        if self.stiffness_for_static_displacement:
            print('\nBEFORE OPTIMIZATION - STIFFNESS')

            print ('stiffness optimization for static displacement target starts\n')

            self.adjust_stiffness_for_target_static_displacement()

        if self.coupling_entries_for_rotation:
            print ('Adjusting coupling entries for 10% target rotation \n')

            self.adjust_coupling_entries_for_target_static_rotation(plot_opt_func = True)

            print ('Finished rotation optimization \n')

# # AFTER optimization
        print('\nAFTER OPTIMIZATION')
        self.model.identify_decoupled_eigenmodes(print_to_console=True)
        print()

# # CAARC FUNCTIONS

    def adjust_CAARC_eigenform_by_factor(self, plot_opt_func = True):

        target = 1.0
        for mode_id in range(1,3): # just for mode 1 and two the third needs an updated comuptation of effectice mass in torque eigenmode
            # ony for each mode 
            #for dof in ['y','z']:#,'a']:
            if mode_id == 1:
                main_dof = 'y'
                remain_dof = ['z','a']
            elif mode_id == 2:
                main_dof = 'z'
                remain_dof = ['y','a']
            initial_Yn = np.copy(self.model.CAARC_eigenmodes['shape'][mode_id][main_dof])

            optimizable_function = partial(self.CAARC_eigenform_factorization_objective_function,
                                            target,
                                            initial_Yn,
                                            mode_id,
                                            main_dof)

            if plot_opt_func:
                plotter_utilities.plot_optimizable_function(optimizable_function, 'Yn ' + ' at mode '+ str(mode_id), np.arange(-5,5,0.01))

            minimization_result = minimize_scalar(optimizable_function,
                                                method='Bounded',
                                                bounds=(1/10000, 1),
                                                options={'disp':False})

            for dof in remain_dof:
                self.model.CAARC_eigenmodes['shape'][mode_id][dof] *= minimization_result.x

            print ('opt factor for dof ' + main_dof +  ' at mode '+str(mode_id), minimization_result.x)
            print (' ')

    def CAARC_eigenform_factorization_objective_function(self, target, initial_Yn, mode_id, main_dof ,multiplier_fctr):

        # need a 2 dimensional function here? --> or just multiply the main direction of the respective mode and use afterwards the same factor for the other direction
        # or call the get_m_eff funtion with main_direction_only = True
        self.model.CAARC_eigenmodes['shape'][mode_id][main_dof] = initial_Yn * multiplier_fctr
        #self.model.CAARC_eigenmodes[str(mode_id)]['z'] = initial_Yn['z'] * multiplier_fctr

        current_denominator = caarc.get_m_eff(self.model.CAARC_eigenmodes, mode_id, True, False)[1]

        result = (current_denominator - target)**2

        return result

# # EXISTING OPTIMIZATION FUNCTIONS
    def adjust_density_for_target_total_mass(self, target_total_mass, print_to_console=False):

        print('BEFORE TUNED DENSITY')
        # calculate to be sure to have most current
        self.model.calculate_total_mass()

        initial_rho = self.model.parameters['rho']

        # using partial to fix some parameters for the
        optimizable_function = partial(self.generic_material_density_objective_function,
                                       target_total_mass,
                                       initial_rho)

        #self.plotter_utilities.plot_optimizable_function(optimizable_function, 'mass', 3)

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/5, 5))
                                              # TODO avoid hardcoding
                                            #   bounds=(1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR))

        #self.plotter_utilities.plot_optimizable_function(optimizable_function, 'mass')

        # returning only one value!
        opt_rho_fctr = minimization_result.x
        self.model.parameters['rho'] = initial_rho * opt_rho_fctr

        

        # re-calculate and print to console
        print('AFTER TUNED DENSITY')
        self.model.calculate_total_mass()
        print()
        if print_to_console:
            print('INITIAL rho:', initial_rho)
            print('OPTIMIZED rho: ', opt_rho_fctr * initial_rho)
            print()

        # re-evaluate
        self.model.calculate_global_matrices()

    def generic_material_density_objective_function(self, target_total_mass, initial_rho, multiplier_fctr):

        for e in self.model.elements:
            e.rho = multiplier_fctr * initial_rho

        # NOTE: do not forget to update G and further dependencies
        self.model.calculate_total_mass()

        return ((self.model.parameters['m_tot']-target_total_mass)**2)

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

    def generic_material_stiffness_objective_function(self, target_freq, target_mode, initial_e, multiplier_fctr):

        for e in self.model.elements:
            e.E = multiplier_fctr * initial_e

            # NOTE: do not forget to update G and further dependencies
            e.evaluate_relative_importance_of_shear()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[target_mode-1]] - target_freq)**2 / target_freq**2

    def adjust_longitudinal_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_a = list(e.A for e in self.model.elements)
        # assuming a linear dependency of shear areas
        initial_a_sy = list(e.Asy for e in self.model.elements)
        initial_a_sz = list(e.Asz for e in self.model.elements)

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

        # NOTE: it seems to need total mass and in general difficult/insensitive to tuning...
        # TODO:
        # self.adjust_density_for_target_total_mass(target_total_mass)

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'longitudinal'

        mode_type_results = self.model.mode_identification_results[identifier]
        # mode_type_results is an ordered list
        m_id = mode_type_results[0]['mode_id']

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[m_id-1]] - target_freq)**2 / target_freq**2

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

        bnds_iy = (1/OptimizableStraightBeam.OPT_FCTR,
                   OptimizableStraightBeam.OPT_FCTR)  # (1/8,8)
        # (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)#(1/15,15)
        bnds_a_sz = (0.4, 1.0)

        minimization_result = minimize(optimizable_function,
                                       init_guess,
                                       method='L-BFGS-B',  # 'SLSQP',#
                                       bounds=(bnds_iy, bnds_a_sz))

        # returning only one value!
        opt_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL iy:', ', '.join([str(val) for val in initial_iy]))
            print()
            print('OPTIMIZED iy: ', ', '.join(
                [str(opt_fctr[0] * val) for val in initial_iy]))
            print('INITIAL a_sz:', ', '.join(
                [str(val) for val in initial_a_sz]))
            print()
            print('OPTIMIZED a_sz: ', ', '.join(
                [str(opt_fctr[1] * val) for val in initial_a_sz]))
            print()
            print('FACTORS: ', ', '.join([str(val) for val in opt_fctr]))
            print()

    def bending_y_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iy, initial_a_sz, multiplier_fctr):

        for e in self.model.elements:
            e.Iy = multiplier_fctr[0] * initial_iy[e.index]
            # assuming a linear dependency of shear areas
            e.Asz = multiplier_fctr[1] * initial_a_sz[e.index]
            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'sway_y'

        mode_type_results = self.model.mode_identification_results[identifier]
        # mode_type_results is an ordered list
        m_id = mode_type_results[0]['mode_id']

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[m_id-1]] - target_freq)**2 / target_freq**2

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

        bnds_iz = (1/OptimizableStraightBeam.OPT_FCTR,
                   OptimizableStraightBeam.OPT_FCTR)  # (1/8,8)
        bnds_a_sy = (1/OptimizableStraightBeam.OPT_FCTR,
                     OptimizableStraightBeam.OPT_FCTR)  # (1/15,15)

        minimization_result = minimize(optimizable_function,
                                       initi_guess,
                                       method='L-BFGS-B',
                                       bounds=(bnds_iz, bnds_a_sy))

        # returning only one value!
        opt_iz_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL iz:', ', '.join([str(val) for val in initial_iz]))
            print()
            print('OPTIMIZED iz: ', ', '.join(
                [str(opt_iz_fctr[0] * val) for val in initial_iz]))
            print()
            print('INITIAL a_sy:', ', '.join(
                [str(val) for val in initial_a_sy]))
            print()
            print('OPTIMIZED a_sy: ', ', '.join(
                [str(opt_iz_fctr[1] * val) for val in initial_a_sy]))
            print('FACTOR: ', opt_iz_fctr)
            print()

    def bending_z_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iz, initial_a_sy, multiplier_fctr):
        
        for e in self.model.elements:
            e.Iz = multiplier_fctr[0] * initial_iz[e.index]
            # assuming a linear dependency of shear areas
            e.Asy = multiplier_fctr[1] * initial_a_sy[e.index]

            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'sway_z'

        mode_type_results = self.model.mode_identification_results[identifier]
        # mode_type_results is an ordered list
        m_id = mode_type_results[0]['mode_id']

        result = (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[m_id-1]] - target_freq)**2 / target_freq**2

        return result

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
            e.It = multiplier_fctr[0] * initial_it[e.index]
            e.Ip = multiplier_fctr[1] * initial_ip[e.index]

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'torsional'

        mode_type_results = self.model.mode_identification_results[identifier]
        # mode_type_results is an ordered list
        m_id = mode_type_results[0]['mode_id']

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[m_id-1]] - target_freq)**2 / target_freq**2

# # Optimizations for coupling of motions 

# # cross stiffness for specified first n nodes 

    def adjust_across_stiffness_lower_section (self, nodes_to_adjust):
        # target: y component of eigenmode 1 of caarc
        # the factorized caarc is used 
        # self.adjust_CAARC_eigenform_by_factor(plot_opt_func=False)
        for e in self.model.elements:
        # initial parameters to adjust 
            if e.index == nodes_to_adjust:
                break
            initial_iz = e.Iz
            initial_a_sy = e.Asy 

            x = e.ReferenceCoords[3] # 2nd node of current element
            target_Yn = caarc.get_CAARC_eigenform_polyfit(self.model.CAARC_eigenmodes, [x])['shape'][1]['y'][0]

            optimizable_function = partial(self.across_stiffness_objective_function,
                                            e.index,
                                            target_Yn,
                                            initial_iz,
                                            initial_a_sy)
            initi_guess = (1.0, 1.0)

            bnds_iz = (1/OptimizableStraightBeam.OPT_FCTR,
                    OptimizableStraightBeam.OPT_FCTR)  # (1/8,8)
            bnds_a_sy = (1/OptimizableStraightBeam.OPT_FCTR,
                        OptimizableStraightBeam.OPT_FCTR)  # (1/15,15)

            minimization_result = minimize(optimizable_function,
                                        initi_guess,
                                        method='L-BFGS-B',
                                        bounds=(bnds_iz, bnds_a_sy),
                                        options={'disp':True})
            
    def across_stiffness_objective_function(self, current_index, target_Yn, initial_iz, initial_a_sy, multiplier_fctr):
        for e in self.model.elements:
            if e.index == current_index:
                e.Iz = multiplier_fctr[0] * initial_iz
                # assuming a linear dependency of shear areas
                e.Asy = multiplier_fctr[1] * initial_a_sy

                # NOTE: do not forget to update further dependencies
                e.evaluate_relative_importance_of_shear()
                #e.evaluate_torsional_inertia()

                # re-evaluate
                self.model.calculate_global_matrices()

                self.model.eigenvalue_solve()

                self.model.decompose_and_quantify_eigenmodes()

                # here just the 1st eigenmode: id 0?
                current_Yn = self.model.decomposed_eigenmodes['values'][0]['y'][current_index]

                result = (current_Yn - target_Yn)**2

        return result 

# # element wise the denisty for a target mode shape

    def adjust_element_density (self, nodes_to_adjust):
        # target: y component of eigenmode 1 of caarc
        # the factorized caarc is used 
        # self.adjust_CAARC_eigenform_by_factor(plot_opt_func=False)
        for e in self.model.elements:
        # initial parameters to adjust 
            if e.index == nodes_to_adjust:
                break
            initial_rho = e.rho

            x = e.ReferenceCoords[3] # 2nd node of current element
            target_Yn = caarc.get_CAARC_eigenform_polyfit(self.model.CAARC_eigenmodes, [x])['shape'][1]['y'][0]

            optimizable_function = partial(self.element_density_objective_function,
                                            e.index,
                                            target_Yn,
                                            initial_rho)

            minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR))

            factor = minimization_result.x

            print ('\nafter density optimization of element ', e.index)
            print ('inital rho: ', initial_rho)
            print ('factor: ', factor)
            print ('opt rho: ', e.rho)
            
    def element_density_objective_function(self, current_index, target_Yn, initial_rho, multiplier_fctr):
        for e in self.model.elements:
            if e.index == current_index:
                e.rho = multiplier_fctr * initial_rho

                # re-evaluate
                self.model.calculate_total_mass()

                self.model.calculate_global_matrices()

                self.model.eigenvalue_solve()

                self.model.decompose_and_quantify_eigenmodes()

                # here just the 1st eigenmode: id 0?
                current_Yn = self.model.decomposed_eigenmodes['values'][0]['y'][current_index]

                result = (current_Yn - target_Yn)**2

        return result 

# # EACH DOF FOR EACH NODE WITH CAARC EIGENFROM AS TARGET (ALL MAX NORMALIZED)

    def adjust_dof_wise_stiffness_for_CAARC_eigenform(self, plot_opt_func = False):
    
        # 1. find the nodes at which target should be evaluated
        node_levels_ParOpt = []
        # TODO: this is on intervals here!!! needs to be at each element (for Beam with 4 elements this is equal in the current model)
        for interval in self.model.parameters['intervals']:
            if interval['bounds'][0] != 0.0: #ground node is skipped 
                node_levels_ParOpt.append(interval['bounds'][0])
        node_levels_ParOpt.append(self.model.parameters['lx']) # add level of last node

        # TODO: so far it only adjusts in the 1st Eigenmode, maybe need other ones to adjust
        for dof in GD.DOF_LABELS[self.model.domain_size][1:4]: # only y,z, alpha
            for node_id, node in enumerate(node_levels_ParOpt):
                # in comp_k ground nodes are already reduced. Thus node_id starts with 0 
                initial_k = self.model.comp_k.copy() 
                mode_id = 1
                
                target_eigenform = caarc.get_CAARC_eigenform_polyfit(self.model.CAARC_eigenmodes, [node])['shape'][mode_id][dof]
                signed_max_disp = get_signed_maximum(caarc.get_CAARC_eigenform_polyfit(self.model.CAARC_eigenmodes)['shape'][mode_id][dof]) 

                # normalization of the current value at current node by maximum value 
                target_disp_max_normalized = target_eigenform[0]/ signed_max_disp 

                optimizable_function = partial(self.CAARC_eigenform_objective_function_dof_wise,
                                               initial_k,
                                               target_disp_max_normalized,
                                               dof,
                                               node_id)
                if plot_opt_func:
                    plotter_utilities.plot_optimizable_function(optimizable_function, 'disp ' + str(dof) + ' at node '+ str(node_id))

                minimization_result = minimize_scalar(optimizable_function,
                                                    method='Bounded',
                                                    bounds=(1/1000, # 100 
                                                            1),
                                                    options={'disp':True})
                
                print ('opt factor for dof ' + str(dof) + ' at node '+str(node_id), minimization_result.x)
                print (' ')

    def CAARC_eigenform_objective_function_dof_wise(self, initial_k, target_disp_max_normalized, dof, current_node_id ,multiplier_fctr):
        
        from source.analysis.eigenvalue_analysis import EigenvalueAnalysis

        
        dof_id = GD.DOF_LABELS[self.model.domain_size].index(dof)
        start_id = GD.DOFS_PER_NODE[self.model.domain_size]  * current_node_id #* GD.NODES_PER_LEVEL
        end_id = GD.DOFS_PER_NODE[self.model.domain_size]

        # design variable: certain entry(ies) in the comp_k
        self.model.comp_k[start_id+dof_id][start_id+dof_id] = initial_k[start_id+dof_id][start_id+dof_id] * multiplier_fctr
        # self.model.comp_k[start_id+dof_id][start_id+dof_id + end_id] = initial_k[start_id+dof_id+ end_id][start_id+dof_id] * multiplier_fctr
        # self.model.comp_k[start_id+dof_id + end_id][start_id+dof_id] = initial_k[start_id+dof_id+ end_id][start_id+dof_id] * multiplier_fctr

        # need eigenvalue_analysis object to get the mass normalized eigenvectors 
        eigenvalue_analysis = EigenvalueAnalysis(self.model, self.eigenvalue_analysis_params_custom)
        eigenvalue_analysis.solve()

        step = GD.DOFS_PER_NODE[self.model.domain_size] #6
        start = GD.DOF_LABELS[self.model.domain_size].index(dof) + step 
        stop = eigenvalue_analysis.eigenform.shape[0] + start - step

        # eigenform is mass normalized
        # need here the displacement of the current node in the current dof direction

        current_eigenform = eigenvalue_analysis.eigenform[start:stop+1:step][:, 0] 
        # 0 ist hier für den ersten Eigenmode, wenn current mode_id an objective function übergeben wird könnte hier auch geloopt werden

        current_signed_max = get_signed_maximum(current_eigenform)
    
        current_disp_max_normalized = current_eigenform[current_node_id] / get_signed_maximum(current_eigenform) 

        result = (current_disp_max_normalized - target_disp_max_normalized)**2
        #result = np.linalg.norm(np.subtract(target_disp_max_normalized, current_disp_max_normalized))

        return result

# # GLOABL STIFFNESS FOR CAARC EIGENFORM --> not sensible: global K changes the frequency but the eigenform will remain the same

    def adjust_global_stiffness_for_CAARC_eigenform(self, plot_opt_func = True):

        initial_k = self.model.comp_k.copy()
        # copy and '=' scheint das gleiche zu sein
        #initial_k = self.model.comp_k
        mode_id = 1
        #initial_k = self.model.k.copy()

        node_ids = []
        node_levels_ParOpt = []
        # TODO: this is on intervals here!!! needs to be at each element if intervals != n_elements
        # --> thus loop over n_elements
        for interval in self.model.parameters['intervals']:
            node_levels_ParOpt.append(interval['bounds'][0])
        node_levels_ParOpt.append(self.model.parameters['lx']) # add level of last node

        for i, level in enumerate(self.model.CAARC_eigenmodes['storey_level']):
            if level in node_levels_ParOpt:
                node_ids.append(i)
        # TODO: has do be done for each dof and the only the according entries in the stiffness matrix must be optimized
        target_disp_vector_by_exact_ids = self.model.CAARC_eigenmodes['1']['y'][node_ids]
        
        target_disp_vector = np.zeros(len(node_levels_ParOpt))
        # here loopable over mode ids and dof labels
        for i, node in enumerate(node_levels_ParOpt):
            target_disp_vector[i] += caarc.get_CAARC_eigenform_polyfit(self.model.CAARC_eigenmodes, node)
        
        
        optimizable_function = partial(self.model.CAARC_eigenform_objective_function,
                                       initial_k,
                                       target_disp_vector)
        #if plot_opt_func:
        #self.plotter_utilities.plot_optimizable_function(optimizable_function, 'disp')

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/1000, # 100 
                                                       1),
                                              options={'disp':True})
        
        print ('opt factor: ', minimization_result.x)

    def CAARC_eigenform_objective_function(self, initial_k, target_disp, multiplier_fctr): 
        
        from source.analysis.eigenvalue_analysis import EigenvalueAnalysis

        #self.model.k = initial_k * multiplier_fctr # comp_k is used in structure_model_eigenvalue_solve?
        # # at this point consider to initialize the element matrices first and then go through the procedure towards the analysis
        self.model.comp_k = initial_k * multiplier_fctr
        #self.model.k = initial_k * multiplier_fctr
        # re evaluated --> comp_k is used in eigenvalue analysis 

        #self.model.calculate_global_matrices()

        eigenvalue_analysis = EigenvalueAnalysis(self.model, self.eigenvalue_analysis_params_custom)
        eigenvalue_analysis.solve()

        #self.model.eigenvalue_solve()

        # for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])), #[0,1,2,...] zip ['x', 'y',...] = [(0, 'x'), (1, 'y'), ...]
        #                       GD.DOF_LABELS[self.structure_model.domain_size]):

        start = GD.DOF_LABELS[self.model.domain_size].index('y')
        step = GD.DOFS_PER_NODE[self.model.domain_size] #6
        stop = eigenvalue_analysis.eigenform.shape[0] + start - step

        # eigenform ohne sckalierung der Eigenvektoren? 
        current_disp_vector = eigenvalue_analysis.eigenform[start:stop+1:step][:,0]

        result = np.linalg.norm(np.subtract(target_disp, current_disp_vector))

        return result

# # SINGLE Y-A COUPLING ENTRIES FOR A STATIC TARGET ROTATION

    def adjust_coupling_entries_for_target_static_rotation(self, plot_opt_func = True):
        '''
        idea: 
        1. perform a static analysis with static force file 
        2. multipliy the rotation alpha with e.g. 1.1
        3. get a multiplier factor that scales the coupling entries of y and alpha
        '''
        from source.analysis.static_analysis import StaticAnalysis

        for node in range(1, self.model.n_nodes): # for every node except the gorund this is done

            initial_k = self.model.k.copy()

            static_analysis = StaticAnalysis(self.model, self.static_analysis_params_custom)
            static_analysis.solve() # here the comp_k is evaluated bevore solving 

            rotations = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('a'):
                                                                    :GD.DOFS_PER_NODE[self.model.domain_size]]
            disp_y = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('y'):
                                                                    :GD.DOFS_PER_NODE[self.model.domain_size]]

            target_ratio = disp_y[node] / (rotations[node] * 1.1)

            if node == 1:
                initial_y = disp_y
                initial_a = rotations
                ratio_initial = disp_y/(rotations)
                ratio_initial[0] = 0.0
                ratio_target = disp_y/(rotations*1.1)
                ratio_target[0] = 0.0
                
            
            optimizable_function = partial(self.static_rotation_stiffness_objective_function,
                                        initial_k,
                                        target_ratio,
                                        node)

            # plot first then minimize!
            if plot_opt_func:
                plotter_utilities.plot_optimizable_function(optimizable_function, 'stiffness at ' + str(node))
                
            minimization_result = minimize_scalar(optimizable_function,
                                                method='Bounded',
                                                bounds=(1/OptimizableStraightBeam.OPT_FCTR, # 100 
                                                        OptimizableStraightBeam.OPT_FCTR),
                                                options={'disp':False})

            print ('optimization factor of node', node, ':', minimization_result.x) 

        # after optimization 

        static_analysis = StaticAnalysis(self.model, self.static_analysis_params_custom)
        static_analysis.solve() 

        rotations = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('a'):
                                                                    :GD.DOFS_PER_NODE[self.model.domain_size]]
        disp_y = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('y'):
                                                                    :GD.DOFS_PER_NODE[self.model.domain_size]]  

        # plot sum comparisons to see what happens 
        fig, axes = plt.subplots(1,4, sharey = True)
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]
        ax1.set_title('rotations alpha')
        
        ax1.plot(initial_a, self.model.nodal_coordinates['x0'], label = 'initial')
        ax1.plot(initial_a*1.1, self.model.nodal_coordinates['x0'], label = 'target')
        ax1.plot(rotations, self.model.nodal_coordinates['x0'], label = 'opt')
        ax1.legend()

        ax2.set_title('displacement y (inital - after)')
        # ax2.plot(initial_y - disp_y, self.model.nodal_coordinates['x0'], label = 'initial')
        # ax2.plot(disp_y, self.model.nodal_coordinates['x0'], label = 'opt')
        ax2.plot(initial_y - disp_y, self.model.nodal_coordinates['x0'], label = 'y')
        ax2.plot(initial_a - rotations, self.model.nodal_coordinates['x0'], label = '\u03B1')
        ax2.legend()

        ax3.set_title('ratios')
        ax3.plot(ratio_initial, self.model.nodal_coordinates['x0'], label = 'initial')
        ax3.plot(ratio_target, self.model.nodal_coordinates['x0'], label = 'target')
        ratio_opt = disp_y/rotations   
        ratio_opt[0] = 0.0
        ax3.plot(ratio_opt, self.model.nodal_coordinates['x0'], label = 'opt')  
        ax3.legend()

        ax4.set_title('target - opt/inital ratio')
        ax4.plot(ratio_target - ratio_opt, self.model.nodal_coordinates['x0'], label = 'opt')
        ax4.plot(ratio_target - ratio_initial, self.model.nodal_coordinates['x0'], label = 'initial')
        ax4.legend()

        ax1.set_ylabel('height [m]')
        plt.show()

    def static_rotation_stiffness_objective_function(self, initial_k, target_ratio, node, multiplier_fctr):
        # 1. find the coupling entries of the current node 
        # id a = 3 , step = 6
        step = GD.DOFS_PER_NODE[self.model.domain_size] 
        row = GD.DOF_LABELS[self.model.domain_size].index('a') + node * step
        column = GD.DOF_LABELS[self.model.domain_size].index('y') + node * step

        self.model.k[row,column] = initial_k[row,column] * multiplier_fctr
        self.model.k[column, row] = initial_k[column, row] * multiplier_fctr

        static_analysis = StaticAnalysis(self.model, self.static_analysis_params_custom)
        static_analysis.solve()

        current_rotation = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('a'):
                                :GD.DOFS_PER_NODE[self.model.domain_size]][node]
        current_dips_y = static_analysis.static_result[GD.DOF_LABELS[self.model.domain_size].index('y'):
                                :GD.DOFS_PER_NODE[self.model.domain_size]][node]

        current_ratio = current_dips_y/current_rotation

        result = (target_ratio[0] - current_ratio[0])**2

        return result 

# # GLOABL STIFFNESS FOR TARGET STATIC DISPLACEMENT VECTOR

    def adjust_stiffness_for_target_static_displacement(self, plot_opt_func = True):
        '''
        idea: 
        1. perform a static analysis with static force file 
        2. multipliy displacement result with e.g. 1.1
        3. get a multiplier factor that scales the whole stiffness matrix to reach the target
        later:
        target could be relation of some lateral displacement to twist 
        '''
        from source.analysis.static_analysis import StaticAnalysis

        initial_k = self.model.k.copy()
        
        static_analysis = StaticAnalysis(self.model, self.static_analysis_params_custom)
        static_analysis.solve()
        target_disp_vector = static_analysis.static_result * 1.1

        # find relevant/non zero tip displacement 
        force_direction = self.static_analysis_params_custom['input']['file_path'][-5]
        tip_disp_index = GD.DOFS_PER_NODE[self.model.domain_size]*(self.model.n_nodes - 1) + \
                                GD.DOF_LABELS[self.model.domain_size].index(force_direction)
        
        target_disp = target_disp_vector[tip_disp_index]

        print ('initial displacement vector: \n')
        print (static_analysis.static_result)
        print ()
        print ('target displacemnet vector: \n')
        print (target_disp_vector)
        print ('\n target tip displacement \n', target_disp[0])
        print()
        
        optimizable_function = partial(self.static_displacment_stiffness_objective_function,
                                       initial_k,
                                       target_disp_vector)

        # plot first then minimize!
        if plot_opt_func:
            plotter_utilities.plot_optimizable_function(optimizable_function, 'stiffness')
            
        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, # 100 
                                                       OptimizableStraightBeam.OPT_FCTR),
                                              options={'disp':True})

        matrix_optimization_factor = minimization_result.x
        print ('optimization factor: ', matrix_optimization_factor)

        #self.plot_optimizable_function(optimizable_function, 'stiffness', matrix_optimization_factor)
        
        # TODO: update the stiffness of the model --> should be used in the static analysis later on
        #self.model.k = initial_k * matrix_optimization_factor
        # re-evaluate is this necessary?
        #self.model.calculate_global_matrices()

    def static_displacment_stiffness_objective_function(self, initial_k, target_disp, multiplier_fctr):
        
        self.model.k = initial_k * multiplier_fctr

        # re - evaluate static result
        static_analysis = StaticAnalysis(self.model, self.static_analysis_params_custom)
        static_analysis.solve()

        current_disp_vector = static_analysis.static_result

        force_direction = self.static_analysis_params_custom['input']['file_path'][-5] # needed since tip diesplacement might be zero
        tip_disp_index = GD.DOFS_PER_NODE[self.model.domain_size]*(self.model.n_nodes - 1) + \
                                GD.DOF_LABELS[self.model.domain_size].index(force_direction)
        
        #print ('current tip displacement: ', current_disp_vector[tip_disp_index])

        current_difference = np.subtract(current_disp_vector, target_disp)
        #print(current_difference)

        result = np.linalg.norm(np.subtract(current_disp_vector, target_disp))#**2
        #result = np.subtract(current_disp_vector, target_disp)**2
        return result #(current_disp - target_disp)**2

# # OLD - - WITH USING A COUPLING PARAMETER NOT AN EXCENTRICITY

    def adjust_coupling_parameter_YT_target_displacement(self, target_disp):
        initial_YT = list(e.YT for e in self.model.elements)

        # using partial to fix some parameters for the
        optimizable_function = partial(self.coupling_torsional_displacementY_objective_function,
                                       target_disp,
                                       initial_YT)
        # --> now objective function is only depent on multiplier_fctr 

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, # 100 
                                                       OptimizableStraightBeam.OPT_FCTR))

        opt_YT_fctr = minimization_result.x

    def coupling_torsional_displacementY_objective_function(self, target_disp, initial_YT, multiplier_fctr):
        for e in self.model.elements:
            e.YT = multiplier_fctr * initial_YT

        # calculate updated stiffness matrix
        self.model.calculate_global_matrices()

        analysis_param = {} # Static analysis just assigns defaults
        static_analysis = StaticAnalysis(self.model, analysis_param)
        static_analysis.solve()

        tip_dof_x_direction = self.model.n_nodes *  GD.DOFS_PER_NODE[self.model.domain_size] - 5
        tip_displacement = static_analysis.static_result[tip_dof_x_direction]

        # static_result or displacment private variable??
        return ((tip_displacement - target_disp)**2)

    def adjust_coupling_parameter_YT_target_freq_sway_y(self, target_freq, target_mode, print_to_console = True):
        initial_YT = list(e.YT for e in self.model.elements)

        # using partial to fix some parameters for the
        optimizable_function = partial(self.coupling_torsional_displacementY_objective_function_target_freq_sway_y,
                                       target_freq,
                                       target_mode,
                                       initial_YT)
        # --> now objective function is only depent on multiplier_fctr 

        minimization_result = minimize_scalar(optimizable_function,
                                              method='Bounded',
                                              bounds=(1/OptimizableStraightBeam.OPT_FCTR, # 100 
                                                       OptimizableStraightBeam.OPT_FCTR))

        # returning only one value!
        opt_fctr = minimization_result.x

        if print_to_console:
            print('optimized YT coupling factor to target frequency')
            print('INITIAL YT:', ', '.join([str(val) for val in initial_YT]))
            print('OPTIMIZED YT: ', ', '.join([str(opt_fctr * val) for val in initial_YT]))
            print('FACTOR', round(opt_fctr, 4))
            print()

    def coupling_torsional_displacementY_objective_function_target_freq_sway_y(self, target_freq, target_mode, initial_YT, multiplier_fctr):
        for e in self.model.elements:
            e.YT = multiplier_fctr * initial_YT[e.index]

        # re-evaluate
        self.model.calculate_global_matrices()

        self.model.eigenvalue_solve()

        self.model.identify_decoupled_eigenmodes()

        identifier = 'sway_y'

        mode_type_results = self.model.mode_identification_results[identifier]
        # mode_type_results is an ordered list
        m_id = mode_type_results[0]['mode_id']

        return (self.model.eig_freqs[self.model.eig_freqs_sorted_indices[m_id-1]] - target_freq)**2 / target_freq**2