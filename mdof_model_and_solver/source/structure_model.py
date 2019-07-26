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
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import math
import matplotlib.pyplot as plt


# TODO: clean up these function, see how to make the shear beam / additional rotational stiffness

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


def evaluate_polynomial(x, coefs):
    val = 0.0
    for idx, coef in enumerate(coefs):
        val += coef * x**idx
    return val


class StraightBeam(object):
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

    AVAILABLE_BCS = ['\"fixed-fixed\"', '\"pinned-pinned\"', '\"fixed-pinned\"',
                     '\"pinned-fixed\"', '\"fixed-free\"', '\"free-fixed\"']

    BC_DOFS = {
        '2D': {'\"fixed-fixed\"': [0, 1, 2, -3, -2, -1],
               '\"pinned-pinned\"': [0, 1, -5, -4],
               '\"fixed-pinned\"': [0, 1, 2, -5, -4],
               '\"pinned-fixed\"': [0, 1, -3, -2, -1],
               '\"fixed-free\"': [0, 1, 2],
               '\"free-fixed\"': [-3, -2, -1]
               },
        '3D': {'\"fixed-fixed\"': [0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1],
               '\"pinned-pinned\"': [0, 1, 2, -6, -5, -4],
               '\"fixed-pinned\"': [0, 1, 2, 3, 4, 5, -6, -5, -4],
               '\"pinned-fixed\"': [0, 1, 2, -6, -5, -4, -3, -2, -1],
               '\"fixed-free\"': [0, 1, 2, 3, 4, 5],
               '\"free-fixed\"': [-6, -5, -4, -3, -2, -1]
               }}

    DOFS_PER_NODE = {'2D': 3,
                     '3D': 6}

    DOF_LABELS = {'2D': ['x', 'y', 'g'],
                  '3D': ['x', 'y', 'z', 'a', 'b', 'g']}

    MODE_CATEGORIZATION = {
        '2D': {
            'longitudinal': ['x'],
            'sway_y': ['z', 'b']},
        '3D': {
            'longitudinal': ['x'],
            'torsional': ['a'],
            'sway_y': ['z', 'b'],
            'sway_z': ['y', 'g']}}

    NODES_PER_LEVEL = 2

    # optimization factor - modify design parameter of scalar function within bounds of
    # 1/OPT_FCTR to OPT_FCTR as multiplier of the parameter
    OPT_FCTR = 5

    THRESHOLD = 1e-8

    def __init__(self, parameters):

        # TODO: add domain size check
        self.domain_size = parameters["model_parameters"]["domain_size"]

        # TODO: include some assign and validate
        # NOTE: for now using the assumption of the prismatic homogenous isotropic beam
        self.parameters = {
            # material
            'rho': parameters["model_parameters"]["system_parameters"]["material"]["density"],
            'e': parameters["model_parameters"]["system_parameters"]["material"]["youngs_modulus"],
            'nu': parameters["model_parameters"]["system_parameters"]["material"]["poisson_ratio"],
            'zeta': parameters["model_parameters"]["system_parameters"]["material"]["damping_ratio"],
            # geometric
            'lx': parameters["model_parameters"]["system_parameters"]["geometry"]["length_x"],
            'n_el': parameters["model_parameters"]["system_parameters"]["geometry"]["number_of_elements"]} 
        
        # defined on intervals as piecewise continous function on an interval starting from 0.0
        self.parameters['intervals'] = []
        for val in parameters["model_parameters"]["system_parameters"]["geometry"]["defined_on_intervals"]:
            self.parameters["intervals"].append({
                'bounds': val['interval_bounds'],
                # further quantities defined by polynomial coefficient as a function of running coord x
                'c_ly': val["length_y"],
                'c_lz': val["length_z"],
                'c_a': val["area"],
                'c_a_sy': val["shear_area_y"],
                'c_a_sz': val["shear_area_z"],
                'c_iy': val["moment_of_inertia_y"],
                'c_iz': val["moment_of_inertia_z"],
                'c_it': val["torsional_moment_of_inertia"]
            })

        # TODO: later probably move to an initalize function
        # material
        # shear modulus
        self.parameters['g'] = self.parameters['e'] / \
            2 / (1+self.parameters['nu'])

        # running coordinate x - in the middle of each beam element
        self.parameters['x'] = [(x + 0.5)/self.parameters['n_el'] * self.parameters['lx']
                                for x in list(range(self.parameters['n_el']))]

        # length of one element - assuming an equidistant grid
        self.parameters['lx_i'] = self.parameters['lx'] / \
            self.parameters['n_el']
        
        # geometric
        # characteristics lengths
        self.parameters['ly'] = [self.evaluate_characteristic_on_interval(
            x, 'c_ly') for x in self.parameters['x']]
        self.parameters['lz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_lz') for x in self.parameters['x']]
      
        # area
        self.parameters['a'] = [self.evaluate_characteristic_on_interval(
            x, 'c_a') for x in self.parameters['x']]
        # effective area of shear
        self.parameters['a_sy'] = [self.evaluate_characteristic_on_interval(
            x, 'c_a_sy') for x in self.parameters['x']]
        self.parameters['a_sz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_a_sz') for x in self.parameters['x']]
        # second moment of inertia
        self.parameters['iy'] = [self.evaluate_characteristic_on_interval(
            x, 'c_iy') for x in self.parameters['x']]
        self.parameters['iz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_iz') for x in self.parameters['x']]
        # torsion constant
        self.parameters['it'] = [self.evaluate_characteristic_on_interval(
            x, 'c_it') for x in self.parameters['x']]

        self.evaluate_torsional_inertia()

        # relative importance of the shear deformation to the bending one
        self.evaluate_relative_importance_of_shear()
        
        length_coords = self.parameters['lx_i'] * \
            np.arange(self.parameters['n_el']+1)

        self.nodal_coordinates = {"x0": length_coords,
                                  # all zeroes as it is being centered and undeformed - user defined center
                                  "y0": np.zeros(len(length_coords)),
                                  "z0": np.zeros(len(length_coords)),
                                  "a0": np.zeros(len(length_coords)),
                                  "b0": np.zeros(len(length_coords)),
                                  "g0": np.zeros(len(length_coords)),
                                  # placeholders for nodal dofs - will be overwritten during analysis
                                  "y": np.zeros(len(length_coords)),
                                  "z": np.zeros(len(length_coords)),
                                  "a": np.zeros(len(length_coords)),
                                  "b": np.zeros(len(length_coords)),
                                  "g": np.zeros(len(length_coords))}

        self.n_nodes = self.parameters['n_el']+1

        # TODO: make BC handling cleaner and compact
        self.all_dofs_global = np.arange(
            self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size])
        bc = '\"' + \
            parameters["model_parameters"]["boundary_conditions"] + '\"'
        if bc in StraightBeam.AVAILABLE_BCS:
            self.bc_dofs = StraightBeam.BC_DOFS[self.domain_size][bc]
        else:
            err_msg = "The BC for input \"" + \
                parameters["model_parameters"]["boundary_conditions"]
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join(StraightBeam.AVAILABLE_BCS)
            raise Exception(err_msg)

        # handle potential elastic BCs
        self.elastic_bc_dofs = {}
        if 'elastic_fixity_dofs' in parameters["model_parameters"]:
            elastic_bc_dofs_tmp = parameters["model_parameters"]["elastic_fixity_dofs"]
        else:
            print(
                'parameters["model_parameters"] does not have "elastic_fixity_dofs"')
            elastic_bc_dofs_tmp = {}
        
        for key in elastic_bc_dofs_tmp:
            # TODO: check if type cast to int is robust enough
            if int(key) not in self.bc_dofs:
                err_msg = "The elastic BC dof for input \"" + key
                err_msg += "\" is not available for " + \
                    parameters["model_parameters"]["boundary_conditions"] + "\n"
                err_msg += "Choose one of: "
                err_msg += ', '.join([str(val) for val in self.bc_dofs])
                raise Exception(err_msg)
            else:
                print('Valid DoF ' + key +
                      ' for elastic constraint, removing from constrained DoFs')
                self.bc_dofs.remove(int(key))

                # updating elastic bc dofs to global numbering with int type
                val = int(key)
                if val < 0:
                    val += len(self.all_dofs_global)

                # add new element
                self.elastic_bc_dofs[val] = elastic_bc_dofs_tmp[key]

        # list copy by slicing -> [:] -> to have a copy by value
        bc_dofs_global = self.bc_dofs[:]
        for idx, dof in enumerate(bc_dofs_global):
            # shift to global numbering the negative values
            if dof < 0:
                bc_dofs_global[idx] = dof + len(self.all_dofs_global)

        # only take bc's of interest
        self.bcs_to_keep = list(set(self.all_dofs_global)-set(bc_dofs_global)) 
        # TODO: AK :is it better to rename it to dof_to_keep than bc_to_keep ??

        self.calculate_global_matrices()

        if 'adapt_for_target_values' in parameters["model_parameters"]:
            print('Found need for adapting structure for target values')
            
            # if a target mass is set, the density will be adjusted, no additional dependencies
            if 'density_for_total_mass' in parameters["model_parameters"]["adapt_for_target_values"]:
                target_total_mass = parameters["model_parameters"]["adapt_for_target_values"]["density_for_total_mass"]
                print('Adapting density for target total mass: ', target_total_mass)

                self.adjust_density_for_target_total_mass(target_total_mass)
            
            # if generically a target mode and frequency is set, the e-modul will be adjusted, g-modul recomputed
            if 'youngs_modulus_for' in parameters["model_parameters"]["adapt_for_target_values"]:
                target_mode = parameters["model_parameters"]["adapt_for_target_values"]["youngs_modulus_for"]["eigenmode"]
                target_freq = parameters["model_parameters"]["adapt_for_target_values"]["youngs_modulus_for"]["eigenfrequency"]
                print('Adapting young\'s modulus for target eigenfrequency: ' +
                      str(target_freq) + ' and mode: ' + str(target_mode))

                self.adjust_e_modul_for_taget_eigenfreq(
                    target_freq, target_mode, True)
                
            if 'geometric_properties_for' in parameters["model_parameters"]["adapt_for_target_values"]:
                
                modes_to_consider = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"]
                modes_possible_to_consider = [*StraightBeam.MODE_CATEGORIZATION[self.domain_size].keys()]
                diff_list = np.setdiff1d(modes_to_consider, modes_possible_to_consider)
                if len(diff_list) != 0:
                    err_msg = "The element(s) \"" + ', '.join(diff_list) +"\"\n" 
                    err_msg += "in provided modes to consider \"" + ', '.join(modes_to_consider) + "\n"
                    err_msg += "\" are not available for consideration\n"
                    err_msg += "Choose one or more of: \""
                    err_msg += ', '.join(modes_possible_to_consider) + "\"\n"
                    raise Exception(err_msg)
                
                # IMPORTANT: a adaptation/tuning/optimization has to be done in the follopwing order 
                #TODO: check dependencies for order of execution and necessery updates

                # 1. LONGITUDINAL
                #TODO: optimize for area -> update rho to match total mass, also update a_sy, a_sz
                if 'longitudinal' in parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"]:
                    identifier = 'longitudinal'

                    id_idx = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"].index(identifier)
                    target_mode = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                    target_freq = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]

                    # update dependencies
                    # for a_sy and a_sz assuming linear dependency with a -> using same scaling factor
                    self.evaluate_relative_importance_of_shear()
                    print()
                    pass

                # 2. and 3. - on of SWAY_Y and/or SWAY_Z
                #TODO: optimize for iz, iy (maybe also extend to a_sy, a_sz -> multi design param opt) -> update ip with new values, also pz, py
                if 'sway_y' in parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"]:
                    identifier = 'sway_y'

                    id_idx = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"].index(identifier)
                    target_mode = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                    target_freq = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]
                    
                    # update dependencies
                    self.evaluate_torsional_inertia()
                    print()
                    pass

                if 'sway_z' in parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"]:
                    identifier = 'sway_z'

                    id_idx = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"].index(identifier)
                    target_mode = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                    target_freq = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]
                    
                    # update dependencies
                    self.evaluate_torsional_inertia()
                    print()
                    pass

                # 4. TORSIONAL
                #TODO: optimize for it -> needs updated model from previous cases
                if 'torsional' in parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"]:
                    identifier = 'torsional'

                    id_idx = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["consider_decomposed_modes"].index(identifier)
                    target_mode = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_mode_ids"][id_idx]
                    target_freq = parameters["model_parameters"]["adapt_for_target_values"]["geometric_properties_for"]["corresponding_eigenfrequencies"][id_idx]
                    
                    print()
                    pass

        else:
            print('No need found for adapting structure for target values')

    def evaluate_relative_importance_of_shear(self, is_bernoulli=False):
        self.parameters['py'] = [12 * self.parameters['e'] * a / (
            self.parameters['g'] * b * self.parameters['lx_i']**2) for a, b in zip(self.parameters['iz'], self.parameters['a_sy'])]
        self.parameters['pz'] = [12 * self.parameters['e'] * a / (
            self.parameters['g'] * b * self.parameters['lx_i']**2) for a, b in zip(self.parameters['iy'], self.parameters['a_sz'])]

        if is_bernoulli:
            # NOTE: Bernoulli beam set to 0.0
            self.parameters['py'] = [0.0 for a,b in zip(self.parameters['iz'], self.parameters['a_sy'])]
            self.parameters['pz'] = [0.0 for a,b in zip(self.parameters['iy'], self.parameters['a_sz'])] 

    def evaluate_torsional_inertia(self):
        # polar moment of inertia
        # assuming equivalency with circle 
        self.parameters['ip'] = [a + b for a,
                                b in zip(self.parameters['iy'], self.parameters['iz'])]

    def calculate_total_mass(self, print_to_console=False):
        self.parameters['m_tot'] = 0.0
        for i in range(len(self.parameters['x'])):
            self.parameters['m_tot'] += self.parameters['a'][i] * \
                self.parameters['rho'] * self.parameters['lx_i']

        if print_to_console:
            print('CURRENT:')
            print('total mass ', self.parameters['m_tot'])
            print('density: ', self.parameters['rho'])
            print()

    def adjust_density_for_target_total_mass(self, target_total_mass):

        # calculate to be sure to have most current
        self.calculate_total_mass(True)

        corr_fctr = target_total_mass / self.parameters['m_tot']

        self.parameters['rho'] *= corr_fctr

        # re-calculate and print to console
        print('AFTER TUNED DENSITY')
        self.calculate_total_mass(True)

    def adjust_e_modul_for_taget_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_e = self.parameters['e']

        # using partial to fix some parameters for the
        optimizable_function = partial(self.stiffness_objective_function,
                                       target_freq, 
                                       target_mode,
                                       initial_e)

        minimization_result = minimize_scalar(optimizable_function,
                                       method='Bounded',
                                       bounds=(1/StraightBeam.OPT_FCTR, StraightBeam.OPT_FCTR))

        # returning only one value!
        opt_e_fctr = minimization_result.x

        if print_to_console:
            print('INITIAL e:', initial_e)
            print('OPTIMIZED e: ', opt_e_fctr * initial_e)
            print()

    def calculate_global_matrices(self):
        # mass matrix
        self.m = self._get_mass()
        # stiffness matrix
        self.k = self._get_stiffness()
        # damping matrix - needs to be done after mass and stiffness as Rayleigh method nees these
        self.b = self._get_damping()

    def stiffness_objective_function(self, target_freq, target_mode, initial_e, multiplier_fctr):

        self.parameters['e'] = multiplier_fctr * initial_e

        # NOTE: do not forget to update G and further dependencies
        self.parameters['g'] = self.parameters['e'] / \
            2 / (1+self.parameters['nu'])
        self.evaluate_relative_importance_of_shear()


        # re-evaluate
        self.calculate_global_matrices()

        self.eigenvalue_solve()

        return (self.eig_freqs[self.eig_freqs_sorted_indices[target_mode-1]] - target_freq)**2 / target_freq**2

    def identify_decoupled_eigenmodes(self, considered_modes=25, print_to_console=False):
        self.eigenvalue_solve()
        
        self.mode_identification_results = {}

        for i in range(considered_modes):
            decomposed_eigenmode = {}
            selected_mode = self.eig_freqs_sorted_indices[i]

            for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.domain_size])),
                                StraightBeam.DOF_LABELS[self.domain_size]):
                start = idx
                step = StraightBeam.DOFS_PER_NODE[self.domain_size]
                stop = self.eigen_modes_raw.shape[0] + idx - step
                decomposed_eigenmode[label] = self.eigen_modes_raw[start:stop +
                                                                            1:step][:, selected_mode]

            for case_id in StraightBeam.MODE_CATEGORIZATION[self.domain_size]:
                match_for_case_id = False

                for dof_contribution_id in StraightBeam.MODE_CATEGORIZATION[self.domain_size][case_id]:
                    if linalg.norm(decomposed_eigenmode[dof_contribution_id]) > StraightBeam.THRESHOLD:
                        match_for_case_id = True
                
                # TODO: check if robust enough for modes where 2 DoFs are involved
                if match_for_case_id:
                    if case_id in self.mode_identification_results:
                        self.mode_identification_results[case_id].append(
                            selected_mode+1)
                    else:
                        self.mode_identification_results[case_id] = [
                            selected_mode+1]

        if print_to_console:
            print('Result of decoupled eigenmode identification for the first ' +
                  str(considered_modes) + ' mode(s)')

            for mode, mode_ids in self.mode_identification_results.items():
                print('  Mode:', mode)
                for mode_id in mode_ids:
                    print('    Eigenform ' + str(mode_id) + ' with eigenfrequency ' + '{:.2f}'.format(
                        self.eig_freqs[self.eig_freqs_sorted_indices[mode_id-1]]) + ' Hz')
            print()
            
    def eigenvalue_solve(self):
        # raw results
        # solving for reduced m and k - applying BCs leads to avoiding rigid body modes
        eig_values_raw, self.eigen_modes_raw = linalg.eigh(
            self.apply_bc_by_reduction(self.k), self.apply_bc_by_reduction(self.m))
        # rad/s
        self.eig_values = np.sqrt(np.real(eig_values_raw))
        self.eig_freqs = self.eig_values / 2. / np.pi
        # sort eigenfrequencies
        self.eig_freqs_sorted_indices = np.argsort(self.eig_freqs)

    def evaluate_characteristic_on_interval(self, running_coord, characteristic_identifier):
        '''
        NOTE: continous polynomial defined within interval
        starting from the local coordinate 0.0
        so a shift is needed
        see shifted coordinate defined as: running_coord-val['bounds'][0]

        TODO: might not be robust enough with end-check -> add some tolerance
        '''
        for val in self.parameters['intervals']:
            if "End" not in val['bounds']:
                if val['bounds'][0] <= running_coord and running_coord < val['bounds'][1]:
                    return evaluate_polynomial(running_coord-val['bounds'][0], val[characteristic_identifier])
            elif "End" in val['bounds']:
                if val['bounds'][0] <= running_coord and running_coord <= self.parameters['lx']:
                    return evaluate_polynomial(running_coord-val['bounds'][0], val[characteristic_identifier])

    def plot_model_properties(self, print_to_console=False):

        if print_to_console:
            print('x: ', ['{:.2f}'.format(x)
                          for x in self.parameters['x']], '\n')
            print('ly: ', ['{:.2f}'.format(x)
                           for x in self.parameters['ly']], '\n')
            print('lz: ', ['{:.2f}'.format(x)
                           for x in self.parameters['lz']], '\n')

            print('a: ', ['{:.2f}'.format(x)
                          for x in self.parameters['a']], '\n')
            print('a_sy: ', ['{:.2f}'.format(x)
                             for x in self.parameters['a_sy']], '\n')
            print('a_sz: ', ['{:.2f}'.format(x)
                             for x in self.parameters['a_sz']], '\n')
            
            print('iy: ', ['{:.2f}'.format(x)
                           for x in self.parameters['iy']], '\n')
            print('iz: ', ['{:.2f}'.format(x)
                           for x in self.parameters['iz']], '\n')
            print('ip: ', ['{:.2f}'.format(x)
                           for x in self.parameters['ip']], '\n')
            print('it: ', ['{:.2f}'.format(x)
                           for x in self.parameters['it']], '\n')

        fig = plt.figure(1)
        plt.plot(self.parameters['x'], self.parameters['a'],
                 'k-', marker='o', label='a')
        plt.plot(self.parameters['x'], self.parameters['a_sy'],
                 'r-', marker='*', label='a_sy')
        plt.plot(self.parameters['x'], self.parameters['a_sz'],
                 'g-', marker='^', label='a_sz')
        plt.legend()
        plt.grid()

        fig = plt.figure(2)
        plt.plot(self.parameters['x'], self.parameters['it'],
                 'k-', marker='o', label='it')
        plt.plot(self.parameters['x'], self.parameters['iy'],
                 'r-', marker='*', label='iy')
        plt.plot(self.parameters['x'], self.parameters['iz'],
                 'g-', marker='^', label='iz')
        plt.plot(self.parameters['x'], self.parameters['ip'],
                 'c-', marker='|', label='ip')
        plt.legend()
        plt.grid()

        fig = plt.figure(3)
        plt.plot(self.parameters['x'], self.parameters['ly'],
                 'r-', marker='*', label='ly')
        plt.plot(self.parameters['x'], self.parameters['lz'],
                 'g-', marker='^', label='lz')
        plt.legend()
        plt.grid()

        fig = plt.figure(4)
        plt.plot(self.parameters['x'], self.parameters['py'],
                 'r-', marker='*', label='py')
        plt.plot(self.parameters['x'], self.parameters['pz'],
                 'g-', marker='^', label='pz')
        plt.legend()
        plt.grid()

        plt.show()

    def apply_bc_by_reduction(self, matrix, axis='both'):
        '''
        list of dofs to apply bc's to provided by self.bc_dofs
        convert prescribed bc's to global dof number
        subtract from all dofs to keep what is needed
        use np.ix_ and ixgrid to extract relevant elements
        '''

        # NOTE: should be quite robust
        # TODO: test
        if axis == 'row':
            rows = len(self.all_dofs_global)
            cols = matrix.shape[1]
            # make a grid of indices on interest
            ixgrid = np.ix_(self.bcs_to_keep, np.arange(matrix.shape[1]))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(self.all_dofs_global)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.bcs_to_keep)
        elif axis == 'both':
            rows = len(self.all_dofs_global)
            cols = rows
            # make a grid of indices on interest
            ixgrid = np.ix_(self.bcs_to_keep, self.bcs_to_keep)
        elif axis == 'row_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.bcs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])
        else:
            err_msg = "The reduction mode with input \"" + axis
            err_msg += "\" for axis is not avaialbe \n"
            err_msg += "Choose one of: \"row\", \"column\", \"both\", \"row_vector\""
            raise Exception(err_msg)

        return matrix[ixgrid]
    
    def recuperate_bc_by_extension(self, matrix, axis='row'):
        '''
        list of dofs to apply the effect of bc 
        by extension
        use np.ix_ and ixgrid to extract relevant elements
        '''

        # NOTE: should be quite robust
        # TODO: test

        # make a grid of indices on interest
        #ixgrid = np.ix_(self.bcs_to_keep, self.bcs_to_keep)

        # create new array with zeros the size it should be
        # with ixgrid take from existing the relevant data and copy to new
        if axis == 'row':
            rows = len(self.all_dofs_global)
            cols = matrix.shape[1]
            # make a grid of indices on interest
            ixgrid = np.ix_(self.bcs_to_keep, np.arange(matrix.shape[1]))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(self.all_dofs_global)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.bcs_to_keep)
        elif axis == 'both':
            rows = len(self.all_dofs_global)
            cols = rows
            # make a grid of indices on interest
            ixgrid = np.ix_(self.bcs_to_keep, self.bcs_to_keep)
        elif axis == 'row_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.bcs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])
        else:
            err_msg = "The extension mode with input \"" + axis
            err_msg += "\" for axis is not avaialbe \n"
            err_msg += "Choose one of: \"row\", \"column\", \"both\", \"row_vector\""
            raise Exception(err_msg)

        extended_matrix = np.zeros((rows, cols))
        # copy the needed element into the extended matrix
        extended_matrix[ixgrid] = matrix

        return extended_matrix

    def _assemble_el_into_glob(self, el_matrix):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        return glob_matrix

    # def __get_el_mass_2D(self):

    #     """
    #     Getting the consistant mass matrix based on analytical integration

    #     USING the consistent mass formulation

    #     mass values for one level
    #     VERSION 3: from Appendix A - Straight Beam Element Matrices - page 228
    #     https://link.springer.com/content/pdf/bbm%3A978-3-319-56493-7%2F1.pdf

    #     Adaptation of the 3D version to 2D
    #     """

    #     m_const = self.parameters['rho'] * self.parameters['a'] * self.parameters['lx_i']

    #     #
    #     # mass values for one level
    #     # define component-wise to have enable better control for various optimization parameters

    #     # axial inertia - along axis x - here marked as x
    #     m_x = m_const / 6.0 
    #     m_x_11 = 2.
    #     m_x_12 = 1.
    #     m_el_x = m_x * np.array([[m_x_11, m_x_12],
    #                              [m_x_12, m_x_11]])

    #     # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
    #     # translation
    #     Py = self.parameters['py']
    #     m_yg = m_const / 210 / (1+Py)**2        
    #     #
    #     m_yg_11 = 70*Py**2 + 147*Py + 78
    #     m_yg_12 = (35*Py**2 + 77*Py + 44) * self.parameters['lx_i'] / 4
    #     m_yg_13 = 35*Py**2 + 63*Py + 27
    #     m_yg_14 = -(35*Py**2 + 63*Py + 26) * self.parameters['lx_i'] / 4
    #     #
    #     m_yg_22 = (7*Py**2 + 14*Py + 8) * self.parameters['lx_i'] **2 / 4
    #     m_yg_23 = - m_yg_14 
    #     m_yg_24 = -(7*Py**2 + 14*Py + 6) * self.parameters['lx_i'] **2 / 4
    #     #
    #     m_yg_33 = m_yg_11
    #     m_yg_34 = -m_yg_12
    #     #
    #     m_yg_44 = m_yg_22
    #     #
    #     m_el_yg_trans = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
    #                                      [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
    #                                      [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
    #                                      [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])
    #     # rotation
    #     m_yg = self.parameters['rho']*self.parameters['iz'] / 30 / (1+Py)**2 / self.parameters['lx_i']        
    #     #
    #     m_yg_11 = 36
    #     m_yg_12 = -(15*Py-3) * self.parameters['lx_i']
    #     m_yg_13 = -m_yg_11
    #     m_yg_14 = m_yg_12
    #     #
    #     m_yg_22 = (10*Py**2 + 5*Py + 4) * self.parameters['lx_i'] **2
    #     m_yg_23 = - m_yg_12
    #     m_yg_24 = (5*Py**2 - 5*Py -1) * self.parameters['lx_i'] **2
    #     #
    #     m_yg_33 = m_yg_11
    #     m_yg_34 = - m_yg_12
    #     #
    #     m_yg_44 = m_yg_22
    #     #
    #     m_el_yg_rot = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
    #                                    [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
    #                                    [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
    #                                    [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

    #     # sum up translation and rotation
    #     m_el_yg = m_el_yg_trans + m_el_yg_rot

    #     # assemble all components
    #     m_el = np.array([[m_el_x[0][0], 0., 0.,                 m_el_x[0][1], 0., 0.],
    #                      [0., m_el_yg[0][0], m_el_yg[0][1],     0., m_el_yg[0][2], m_el_yg[0][3]],
    #                      [0., m_el_yg[0][1], m_el_yg[1][1],     0., m_el_yg[1][2], m_el_yg[1][3]],
                         
    #                      [m_el_x[1][0], 0., 0.,                 m_el_x[1][1], 0., 0.],
    #                      [0., m_el_yg[0][2], m_el_yg[1][2],     0., m_el_yg[2][2], m_el_yg[2][3]],
    #                      [0., m_el_yg[0][3], m_el_yg[1][3],     0., m_el_yg[2][3], m_el_yg[3][3]]])

    #     return m_el

    def __get_el_mass_3D(self, i):
        """
        Getting the consistant mass matrix based on analytical integration

        USING the consistent mass formulation

        mass values for one level
        VERSION 3: from Appendix A - Straight Beam Element Matrices - page 228
        https://link.springer.com/content/pdf/bbm%3A978-3-319-56493-7%2F1.pdf
        """

        m_const = self.parameters['rho'] * \
            self.parameters['a'][i] * self.parameters['lx_i']

        #
        # mass values for one level
        # define component-wise to have enable better control for various optimization parameters

        # axial inertia - along axis x - here marked as x
        m_x = m_const / 6.0 
        m_x_11 = 2.
        m_x_12 = 1.
        m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                 [m_x_12, m_x_11]])
        # torsion inertia - around axis x - here marked as alpha - a
        m_a = m_const * self.parameters['ip'][i]/self.parameters['a'][i] / 6.0 
        m_a_11 = 2
        m_a_12 = 1
        m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                 [m_a_12, m_a_11]])

        # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
        # translation
        Py = self.parameters['py'][i]
        m_yg = m_const / 210 / (1+Py)**2        
        #
        m_yg_11 = 70*Py**2 + 147*Py + 78
        m_yg_12 = (35*Py**2 + 77*Py + 44) * self.parameters['lx_i'] / 4
        m_yg_13 = 35*Py**2 + 63*Py + 27
        m_yg_14 = -(35*Py**2 + 63*Py + 26) * self.parameters['lx_i'] / 4
        #
        m_yg_22 = (7*Py**2 + 14*Py + 8) * self.parameters['lx_i'] ** 2 / 4
        m_yg_23 = - m_yg_14 
        m_yg_24 = -(7*Py**2 + 14*Py + 6) * self.parameters['lx_i'] ** 2 / 4
        #
        m_yg_33 = m_yg_11
        m_yg_34 = -m_yg_12
        #
        m_yg_44 = m_yg_22
        #
        m_el_yg_trans = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                         [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                         [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                         [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])
        # rotation
        m_yg = self.parameters['rho']*self.parameters['iz'][i] / \
            30 / (1+Py)**2 / self.parameters['lx_i']
        #
        m_yg_11 = 36
        m_yg_12 = -(15*Py-3) * self.parameters['lx_i']
        m_yg_13 = -m_yg_11
        m_yg_14 = m_yg_12
        #
        m_yg_22 = (10*Py**2 + 5*Py + 4) * self.parameters['lx_i'] ** 2
        m_yg_23 = - m_yg_12
        m_yg_24 = (5*Py**2 - 5*Py - 1) * self.parameters['lx_i'] ** 2
        #
        m_yg_33 = m_yg_11
        m_yg_34 = - m_yg_12
        #
        m_yg_44 = m_yg_22
        #
        m_el_yg_rot = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                       [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                       [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                       [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

        # sum up translation and rotation
        m_el_yg = m_el_yg_trans + m_el_yg_rot

        # bending - inertia along axis z, rotations around axis y - here marked as beta - b
        # translation
        Pz = self.parameters['pz'][i]
        m_zb = m_const / 210 / (1+Pz)**2        
        #
        m_zb_11 = 70*Pz**2 + 147*Pz + 78
        m_zb_12 = -(35*Pz**2 + 77*Pz + 44) * self.parameters['lx_i'] / 4
        m_zb_13 = 35*Pz**2 + 63*Pz + 27
        m_zb_14 = (35*Pz**2 + 63*Pz + 26) * self.parameters['lx_i'] / 4
        #
        m_zb_22 = (7*Pz**2 + 14*Pz + 8) * self.parameters['lx_i'] ** 2 / 4
        m_zb_23 = -m_zb_14
        m_zb_24 = -(7*Pz**2 + 14*Pz + 6) * self.parameters['lx_i'] ** 2 / 4
        #
        m_zb_33 = m_zb_11
        m_zb_34 = - m_zb_12
        #
        m_zb_44 = m_zb_22
        #
        m_el_zb_trans = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                         [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                         [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                         [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
        # rotation
        m_zb = self.parameters['rho']*self.parameters['iy'][i] / \
            30 / (1+Pz)**2 / self.parameters['lx_i']
        #
        m_zb_11 = 36
        m_zb_12 = (15*Pz-3) * self.parameters['lx_i']
        m_zb_13 = -m_zb_11
        m_zb_14 = m_zb_12
        #
        m_zb_22 = (10*Pz**2 + 5*Pz + 4) * self.parameters['lx_i'] ** 2
        m_zb_23 = -m_zb_12
        m_zb_24 = (5*Pz**2 - 5*Pz - 1) * self.parameters['lx_i'] ** 2
        #
        m_zb_33 = m_zb_11
        m_zb_34 = -m_zb_12
        #
        m_zb_44 = m_zb_22
        #
        m_el_zb_rot = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                       [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                       [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                       [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
        
        # sum up translation and rotation
        m_el_zb = m_el_zb_trans + m_el_zb_rot

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

        return m_el

    def _get_mass(self):

        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.__get_el_mass_3D(i)
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        return glob_matrix

        # if self.domain_size == '2D':
        #     m_el = self.__get_el_mass_2D()
        # elif self.domain_size == '3D':
        #     m_el = self.__get_el_mass_3D()
        # else:
        #     pass

        # # return stiffness matrix
        # return self._assemble_el_into_glob(m_el)

    # def __get_el_stiffness_2D(self):
        
    #     """
    #     stiffness values for one level
    #     VERSION 2
        
    #     NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
    #     seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
    #     implemented mass matrices similar to the stiffness one
        
    #     Adaptation of the 3D version to 2D
    #     """

    #     # axial stiffness - along axis x - here marked as x
    #     k_x = self.parameters['e']*self.parameters['a']/self.parameters['lx_i']
    #     k_x_11 = 1.0
    #     k_x_12 = -1.0
    #     k_el_x = k_x * np.array([[k_x_11, k_x_12],
    #                              [k_x_12, k_x_11]])
        
    #     # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
    #     beta_yg = self.parameters['py']
    #     k_yg = self.parameters['e']*self.parameters['iz']/(1+beta_yg)/self.parameters['lx_i']**3
    #     #
    #     k_yg_11 = 12.
    #     k_yg_12 = 6. * self.parameters['lx_i']
    #     k_yg_13 = -k_yg_11
    #     k_yg_14 = k_yg_12
    #     #
    #     k_yg_22 = (4.+beta_yg) * self.parameters['lx_i'] **2
    #     k_yg_23 = -k_yg_12
    #     k_yg_24 = (2-beta_yg) * self.parameters['lx_i'] ** 2
    #     #
    #     k_yg_33 = k_yg_11
    #     k_yg_34 = -k_yg_12
    #     #
    #     k_yg_44 = k_yg_22
    #     #
    #     k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
    #                                [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
    #                                [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
    #                                [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

    #     # assemble all components
    #     k_el = np.array([[k_el_x[0][0], 0., 0.,                 k_el_x[0][1], 0., 0.],
    #                      [0., k_el_yg[0][0], k_el_yg[0][1],     0., k_el_yg[0][2], k_el_yg[0][3]],       
    #                      [0., k_el_yg[0][1], k_el_yg[1][1],     0., k_el_yg[1][2], k_el_yg[1][3]],
                         
    #                      [k_el_x[1][0], 0., 0.,                 k_el_x[1][1], 0., 0.],
    #                      [0., k_el_yg[0][2], k_el_yg[1][2],     0., k_el_yg[2][2], k_el_yg[2][3]],
    #                      [0., k_el_yg[0][3], k_el_yg[1][3],     0., k_el_yg[2][3], k_el_yg[3][3]]])

    #     return k_el

    def __get_el_stiffness_3D(self, i):
        """
        stiffness values for one level
        VERSION 2
        
        NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
        seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
        implemented mass matrices similar to the stiffness one
        """

        # axial stiffness - along axis x - here marked as x
        k_x = self.parameters['e'] * \
            self.parameters['a'][i]/self.parameters['lx_i']
        k_x_11 = 1.0
        k_x_12 = -1.0
        k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                 [k_x_12, k_x_11]])
        # torsion stiffness - around axis x - here marked as alpha - a
        k_a = self.parameters['g'] * \
            self.parameters['it'][i]/self.parameters['lx_i']
        k_a_11 = 1.0
        k_a_12 = -1.0
        k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                 [k_a_12, k_a_11]])
        # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
        beta_yg = self.parameters['py'][i]
        k_yg = self.parameters['e']*self.parameters['iz'][i] / \
            (1+beta_yg)/self.parameters['lx_i']**3
        #
        k_yg_11 = 12.
        k_yg_12 = 6. * self.parameters['lx_i']
        k_yg_13 = -k_yg_11
        k_yg_14 = k_yg_12
        #
        k_yg_22 = (4.+beta_yg) * self.parameters['lx_i'] ** 2
        k_yg_23 = -k_yg_12
        k_yg_24 = (2-beta_yg) * self.parameters['lx_i'] ** 2
        #
        k_yg_33 = k_yg_11
        k_yg_34 = -k_yg_12
        #
        k_yg_44 = k_yg_22
        #
        k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
                                   [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
                                   [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
                                   [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

        # bending - displacement along axis z, rotations around axis y - here marked as beta - b
        beta_zb = self.parameters['pz'][i]
        k_zb = self.parameters['e']*self.parameters['iy'][i] / \
            (1+beta_zb)/self.parameters['lx_i']**3
        #
        k_zb_11 = 12.
        k_zb_12 = -6. * self.parameters['lx_i']
        k_zb_13 = -12.
        k_zb_14 = k_zb_12
        #
        k_zb_22 = (4.+beta_zb) * self.parameters['lx_i'] ** 2
        k_zb_23 = -k_zb_12 
        k_zb_24 = (2-beta_zb) * self.parameters['lx_i'] ** 2
        #
        k_zb_33 = k_zb_11
        k_zb_34 = - k_zb_12
        #
        k_zb_44 = k_zb_22
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

        return k_el

    def _get_stiffness(self):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.__get_el_stiffness_3D(i)
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        
        if self.elastic_bc_dofs is not None:
            print('Applying elastic constraint values to stiffness matrix')
            for key in self.elastic_bc_dofs:
                print('Stiffness value for global dof ' + str(key) +
                      ' before update: ' + str(glob_matrix[key, key]))
                glob_matrix[key, key] = glob_matrix[key, key] + \
                    self.elastic_bc_dofs[key]
                print('Stiffness value for global dof ' + str(key) +
                      ' after update: ' + str(glob_matrix[key, key]))
                print()
  
        return glob_matrix

        # if self.domain_size == '2D':
        #     k_el = self.__get_el_stiffness_2D()
        # elif self.domain_size == '3D':
        #     k_el = self.__get_el_stiffness_3D()
        # else:
        #     pass

        # # return stiffness matrix
        # return self._assemble_el_into_glob(k_el)

    def _get_damping(self):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """
        print("Calculating damping b in MDoFMixedModel derived class \n")
        mode_i = 0
        mode_j = 1
        zeta_i = self.parameters['zeta']
        zeta_j = zeta_i

        self.eigenvalue_solve()

        a = np.linalg.solve(0.5 *
                            np.array(
                                [[1 / self.eig_values[self.eig_freqs_sorted_indices[mode_i]],
                                  self.eig_values[
                                  self.eig_freqs_sorted_indices[mode_i]]],
                                    [1 / self.eig_values[self.eig_freqs_sorted_indices[mode_j]],
                                     self.eig_values[
                                     self.eig_freqs_sorted_indices[
                                         mode_j]]]]),
                            [zeta_i, zeta_j])

        # return back the whole matrix - without BCs applied
        return a[0] * self.m + a[1] * self.k


'''
NOTE:
For now backup code to see how tuning/optimization could work
Remove once finished
'''


# class MDoFMixed2DModel(SimplifiedCantileverStructure):
#     """
#     A multi-degree-of-freedom MDoF model assuming
#     bending-type deformations using the Euler-Bernoulli
#     beam theory.

#     ATTENTION:
#     For this model a homogenous distribution of mass,
#     stiffness and damping is a premise. For other cases
#     this model is not adequate and changes need to be done.

#     """

#     def __init__(self,
#                  rho=1.,
#                  area=1.,
#                  target_freq=1.,
#                  target_mode=1,
#                  zeta=0.05,
#                  level_height=1,
#                  num_of_levels=10,
#                  gamma=0.5,
#                  name="DefaultMDoF2DMixedModel"):

#         self.gamma = gamma

#         StraightBeam.DOFS_PER_NODE[self.domain_size] = 2
#         StraightBeam.NODES_PER_LEVEL = 2

#         m = self._calculate_mass(rho, area, level_height, num_of_levels)
#         k = self._calculate_stiffness(
#             m, level_height, num_of_levels, target_freq, target_mode)
#         b = self._calculate_damping(m, k, zeta)

#         height_coordinates = self._get_nodal_coordinates(
#             level_height, num_of_levels)

#         nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
#                              "y0": height_coordinates,
#                              "x": None,
#                              "y": None}

#         super().__init__(m, b, k, nodal_coordinates, name, category='MDoF2DMixed')

#     def _get_nodal_coordinates(self, level_height, num_of_levels):
#         nodal_coordinates = level_height * np.arange(1, num_of_levels+1)
#         return nodal_coordinates

#     def _calculate_mass(self, rho, area, level_height, num_of_levels, is_lumped=False):
#         """
#         Getting the consistant mass matrix based on analytical integration
#         """
#         # mass values for one level
#         length = level_height
#         m_const = rho * area * length

#         if not is_lumped:
#             # case of consistent mass matrix
#             m_beam_elem = 1/420 * np.array([[156, 22*length, 54,   -13*length],
#                                             [22*length, 4*length**2,
#                                                 13*length, -3*length**2],
#                                             [54, 13*length, 156, -22*length],
#                                             [-13*length, -3*length**2, -22*length, 4*length**2]])

#             # NOTE: for now 1/175 on diagonal for rotational inertia otherwise problems with eigenvalue solve
#             # 1/175 seems to be the lowest value for pure shear to still deliver the correct eigenfreq
#             m_shear_elem = 1/6 * np.array([[2,  0,  1,  0],
#                                            [0,  1/175, 0,  0],
#                                            [1,  0,  2,  0],
#                                            [0,  0,  0,  1/175]])

#         else:
#             # case of lumped mass matrix
#             alpha = 1/50  # generally between 0 and 1/50
#             m_beam_elem = np.array([[1/2, 0,                 0,   0],
#                                     [0,   alpha * length**2, 0,   0],
#                                     [0,   0,                 1/2, 0],
#                                     [0,   0,                 0,   alpha * length ** 2]])

#             # NOTE: for now 1/2500 on diagonal for rotational inertia otherwise problems with eigenvalue solve
#             # 1/2500 seems to be the lowest value for pure shear to still deliver the correct eigenfreq
#             m_shear_elem = np.array([[1/2, 0,   0,   0],
#                                      [0,   1/2500,   0,   0],
#                                      [0,   0,   1/2, 0],
#                                      [0,   0,   0,   1/2500]])

#         # global mass matrix initialization with zeros
#         m_glob = np.zeros((2 * num_of_levels + 2, 2 * num_of_levels + 2))

#         # shape function values for superposition of bending and shear
#         sf_b_val = (1-shape_function_exp(shift_normalize(
#                     map_lin_to_log(self.gamma))))
#         sf_s_val = shape_function_exp(shift_normalize(
#             map_lin_to_log(self.gamma)))

#         # fill global mass matrix entries
#         for i in range(num_of_levels):
#             m_temp = np.zeros(
#                 (2 * num_of_levels + 2, 2 * num_of_levels + 2))
#             m_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = sf_b_val * \
#                 m_beam_elem + sf_s_val * m_shear_elem
#             m_glob += m_const * m_temp

#         # remove the fixed degrees of freedom
#         # at first node all dofs are considered fixed for the cantilever beam
#         # generate a list of indices 0, 1, 2,...StraightBeam.DOFS_PER_NODE[self.domain_size]
#         # go through it in reverse order to keep the numbering intact
#         for dof in range(StraightBeam.DOFS_PER_NODE[self.domain_size])[::-1]:
#             # delete corresponding row
#             m_glob = np.delete(m_glob, dof, axis=0)
#             # delet corresponding column
#             m_glob = np.delete(m_glob, dof, axis=1)

#         # return stiffness matrix
#         return m_glob

#     def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
#         """
#         Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all
#         the elements and the value is calculated using an optimization (or "tuning")
#         for a target frequency of a target mode.
#         """
#         print("Calculating stiffness k in MDoFMixedModel derived class \n")

#         # setup k_scalar_guess as the input for the standard k for a shear-type
#         # MDoF
#         # guess stiffness values for shear and bending
#         # as initial value for optimization

#         k_scalar_g = 1000.

#         # using partial to fix some parameters for the
#         optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar,
#                                        m,
#                                        level_height,
#                                        num_of_levels,
#                                        target_freq,
#                                        target_mode)

#         print("Optimization for the target k matrix in MDoFMixedModel \n")
#         minimization_result = minimize(optimizable_function,
#                                        k_scalar_g, method='Powell',
#                                        options={'disp': True})

#         # returning only one value!
#         k_scalar_opt = minimization_result.x
#         print("K scalar optimized: ", k_scalar_opt, '\n')

#         return self._assemble_k(level_height, num_of_levels, k_scalar_opt)

#     def _assemble_k(self, level_height, num_of_levels, k_scalar, magnitude_difference=CUST_MAGNITUDE, beta=0.0):

#         # pure bending
#         # gamma = 1.0
#         # pure shear
#         # gamma = 0.0
#         """
#         For the MDoFMixed model stiffness distribution according to beam theory is assumed
#         the stiffness matrix is asembled with the k_scalar calculated.
#         """

#         # element length / level height
#         length = level_height

#         # beam stiffness
#         k_beam = k_scalar

#         # shear (spring) stiffness
#         k_shear = k_scalar

#         # stifness values for one level
#         # TODO: check as it seems Timoschenko beam needs some additional off-diagonal terms to be affected by beta
#         # similar as in 3D
#         k_beam_elem = 1/(1+beta) * np.array([[12,     6 * length,         -12,     6 * length],
#                                              [6 * length, (4+beta) * length **
#                                               2, -6 * length, 2 * length ** 2],
#                                              [-12,    -6 * length,
#                                                  12,    -6 * length],
#                                              [6 * length, 2 * length ** 2, -6 * length, (4+beta) * length ** 2]])

#         shear_rotation_magnitude = magnitude(
#             (4+beta) * length ** 2) + magnitude_difference

#         k_shear_elem = np.array([[12, 0, -12, 0],
#                                  [0, (4+beta) * length **
#                                   2 * 10**shear_rotation_magnitude, 0, 0],
#                                  [-12, 0, 12, 0],
#                                  [0, 0, 0, (4+beta) * length **
#                                   2 * 10**shear_rotation_magnitude]])

#         # global stiffness matrix initialization with zeros
#         k_glob = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                            (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))

#         # shape function values for superposition of bending and shear
#         sf_b_val = (1-shape_function_exp(shift_normalize(
#                     map_lin_to_log(self.gamma))))
#         sf_s_val = shape_function_exp(shift_normalize(
#             map_lin_to_log(self.gamma)))

#         print('Gamma bending 1.0')
#         print('Gamma shear 0.0')
#         print('Gamma prescribed ', self.gamma, '\n')

#         print('S(hape) f(unction) values for')
#         print('Shear ', sf_s_val)
#         print('Bend ', sf_b_val)
#         print('\n')

#         # fill global stiffness matix entries
#         for i in range(num_of_levels):
#             k_temp = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                                (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))

#             # beam part
#             k_temp[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
#                    StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] = sf_b_val * k_beam * k_beam_elem
#             # shear part

#             k_temp[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
#                    StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += sf_s_val * k_shear * k_shear_elem

#             k_glob += k_temp

#         # remove the fixed degrees of freedom
#         # at first node all dofs are considered fixed for the cantilever beam
#         # generate a list of indices 0, 1, 2,...StraightBeam.DOFS_PER_NODE[self.domain_size]
#         # go through it in reverse order to keep the numbering intact
#         for dof in range(StraightBeam.DOFS_PER_NODE[self.domain_size])[::-1]:
#             # delete corresponding row
#             k_glob = np.delete(k_glob, dof, axis=0)
#             # delet corresponding column
#             k_glob = np.delete(k_glob, dof, axis=1)

#         # return stiffness matrix
#         return k_glob

#     def _calculate_damping(self, m, k, zeta):
#         """
#         Calculate damping b based upon the Rayleigh assumption
#         using the first 2 eigemodes - here generically i and i
#         """
#         print("Calculating damping b in MDoFMixedModel derived class \n")
#         mode_i = 0
#         mode_j = 1
#         zeta_i = zeta
#         zeta_j = zeta

#         # TODO: try to avoid this code duplication

#         # # https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
#         # print(" If mass is positive definite should return True")
#         # print(np.all(np.linalg.eigvals(m) > 0))
#         # wait = input("check...")

#         # raw results
#         eig_values_raw, eigen_modes_raw = linalg.eigh(self.apply_bc_by_reduction(k), self.apply_bc_by_reduction(m))
#         # rad/s
#         eig_values = np.sqrt(np.real(eig_values_raw))
#         # 1/s = Hz
#         eig_freqs = eig_values / 2. / np.pi
#         # sort eigenfrequencies
#         eig_freqs_sorted_indices = np.argsort(eig_freqs)
#         #

#         a = np.linalg.solve(0.5 *
#                             np.array(
#                                 [[1 / eig_values[eig_freqs_sorted_indices[mode_i]],
#                                   eig_values[
#                                   eig_freqs_sorted_indices[mode_i]]],
#                                     [1 / eig_values[eig_freqs_sorted_indices[mode_j]],
#                                      eig_values[
#                                      eig_freqs_sorted_indices[
#                                          mode_j]]]]),
#                             [zeta_i, zeta_j])
#         return a[0] * m + a[1] * k

#     # k_scalar, k_beam):
#     def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
#         k = self._assemble_k(level_height, num_of_levels,
#                              k_scalar)

#         # TODO: try to avoid this code duplication
#         # raw results
#         eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
#         # rad/s
#         eig_values = np.sqrt(np.real(eig_values_raw))
#         # 1/s = Hz
#         eig_freqs = eig_values / 2. / np.pi
#         # sort eigenfrequencies
#         eig_freqs_sorted_indices = np.argsort(eig_freqs)
#         #

#         current_target_freq = eig_freqs[eig_freqs_sorted_indices[target_mode-1]]

#         return (target_freq - current_target_freq) ** 2 / target_freq**2

#     def _load_distribution(self, level_height, num_of_levels, wind_forces):

#         # Nodal distribution the applis load on a beam model
#         def nodal_force(force):
#             F = [force / 2, force * level_height / 12,
#                  force / 2, -force * level_height / 12]
#             return F

#         nodal_load = list(map(nodal_force, wind_forces))

#         load = np.zeros(2 * num_of_levels + 2)

#         for i in range(num_of_levels):
#             load_temp = np.zeros(2 * num_of_levels + 2)
#             load_temp[2 * i:2 * i + 4] = nodal_load[i]
#             load += load_temp

#         # remove the fixed degrees of freedom
#         rdof = [1, 0]
#         for dof in rdof:
#             load = np.delete(load, dof, axis=0)

#         return load

# # NOTE: for now only a copy of the 2D - WIP 

# class MDoFMixed3DModel(SimplifiedCantileverStructure):
#     """
#     A 2D/3D prismatic homogenous isotopic Timoshenko beam element
#     Including shear and rotationary inertia and deformation
#     Using a consistent mass formulation

#     Definition of axes:
#         1. Longitudinal axis: x with rotation alpha around x
#         2. Transversal axes: 
#             y with rotation beta around y 
#             z with rotation gamma around z

#     Degrees of freedom DoFs
#         1. 2D: displacements x, y, rotation g(amma) around z 
#             -> element DoFs for nodes i and j of one element
#                 [0, 1, 2, 3, 4, 5, 6] = [x_i, y_i, g_i, 
#                                          x_j, y_j, g_j]

#         2. 3D: displacements x, y, z, rotationS a(lpha), b(eta), g(amma) 
#             -> element DoFs for nodes i and j of one element
#                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = [x_i, y_i, z_i, a_i, b_i, g_i,
#                                                           x_j, y_j, z_j, a_j, b_j, g_j]

#     TODO:
#         1. add a parametrization to include artificial (and possibly also local) 
#             incrementation of stiffness and mass (towards a shear beam and/or point mass/stiffness)
#         2. add a parametrization for tunig to eigen frequencies and mode shapes
#         3. add a parametrization to be able to specify zones with altering mass ditribution
#         4. add a parametrization to be able to handle a certain distribution or area or similar along length
#         5. extend to be able to handle non-centric centre of elasticity with respect to center of mass
#     """

#     def __init__(self,
#                  rho=1.,
#                  area=1.,
#                  target_freq=1.,
#                  target_mode=1,
#                  zeta=0.05,
#                  level_height=1,
#                  num_of_levels=10,
#                  gamma=0.5,
#                  name="DefaultMDoF3DMixedModel"):

#         self.gamma = gamma

#         StraightBeam.DOFS_PER_NODE[self.domain_size] = 6
#         StraightBeam.NODES_PER_LEVEL = 2

#         # TODO: implement bc type for 2d, apply not in the matrix build
#         # pass to constructor
#         bc='fixed-fixed'
#         #bc='fixed-free'
#         #bc='free-fixed'
#         #bc='pinned-pinned'
#         #bc='fixed-pinned'
#         #bc='pinned-fixed'
#         if bc == 'fixed-fixed':
#             self.bc_dofs = [0,1,2,3,4,5,-6,-5,-4,-3,-2,-1]
#         elif bc == 'pinned-pinned':
#             self.bc_dofs = [0,1,2,-6,-5,-4]
#         elif bc == 'fixed-pinned':
#             self.bc_dofs = [0,1,2,3,4,5,-6,-5,-4]
#         elif bc == 'pinned-fixed':
#             self.bc_dofs = [0,1,2,-6,-5,-4,-3,-2,-1]
#         elif bc == 'fixed-free':
#             self.bc_dofs = [0,1,2,3,4,5]
#         elif bc == 'free-fixed':
#             self.bc_dofs = [-6,-5,-4,-3,-2,-1]
#         else:
#             err_msg =  "The BC for input \"" + bc
#             err_msg +=  "\" is not available \n"
#             err_msg += "Choose one of: "
#             err_msg += ', '.join(['\"fixed-fixed\"', '\"pinned-pinned\"', '\"fixed-pinned\"', 
#                                    '\"pinned-fixed\"', '\"fixed-free\"', '\"free-fixed\"'])
#             raise Exception(err_msg)

#         self.all_dofs_global = np.arange(0, (num_of_levels+1)*StraightBeam.DOFS_PER_NODE[self.domain_size])

#         m = self._calculate_mass(rho, area, level_height, num_of_levels)
#         k = self._calculate_stiffness(
#             m, level_height, num_of_levels, target_freq, target_mode)
#         b = self._calculate_damping(m, k, zeta)

#         height_coordinates = self._get_nodal_coordinates(
#             level_height, num_of_levels)


#         nodal_coordinates = {"x0": np.zeros(len(height_coordinates)),
#                              "y0": height_coordinates,
#                              "x": None,
#                              "y": None}

#         super().__init__(m, b, k, nodal_coordinates, name, category='MDoF3DMixed')
#         # TODO: to move bcs to constructor
#         # super().__init__(m, b, k, bc_dofs, nodal_coordinates, name, category='MDoF3DMixed')

#     def apply_bc_by_reduction(self, matrix):
#         '''
#         list of dofs to apply bc's to provided by self.bc_dofs
#         convert prescribed bc's to global dof number
#         subtract from all dofs to keep what is needed
#         use np.ix_ and ixgrid to extract relevant elements
#         '''

#         # NOTE: should be quite robust
#         # TODO: test

#         # list copy by slicing -> [:] -> to have a copy by value
#         bc_dofs_global = self.bc_dofs[:]
#         for idx, dof in enumerate(bc_dofs_global):
#             # shift to global numbering the negative values
#             if dof < 0:
#                 bc_dofs_global[idx] = dof + (len(self.all_dofs_global)+1)

#         # only take bc's of interes
#         bcs_to_keep = list(set(self.all_dofs_global)-set(bc_dofs_global))

#         # make a grid of indices on interest
#         ixgrid = np.ix_(bcs_to_keep, bcs_to_keep) 

#         return matrix[ixgrid]

#     def recuperate_bc_by_extension(self, matrix,axis='row'):
#         '''
#         list of dofs to apply the effect of bc 
#         by extension
#         use np.ix_ and ixgrid to extract relevant elements
#         '''

#         # NOTE: should be quite robust
#         # TODO: test


#         # list copy by slicing -> [:] -> to have a copy by value
#         bc_dofs_global = self.bc_dofs[:]
#         for idx, dof in enumerate(bc_dofs_global):
#             # shift to global numbering the negative values
#             if dof < 0:
#                 bc_dofs_global[idx] = dof + (len(self.all_dofs_global)+1)

#         # only take bc's of interes
#         bcs_to_keep = list(set(self.all_dofs_global)-set(bc_dofs_global))

#         # make a grid of indices on interest
#         ixgrid = np.ix_(bcs_to_keep, bcs_to_keep) 

#         # create new array with zeros the size it should be
#         # with ixgrid take from existing the relevant data and copy to new
#         if axis == 'row':
#             rows = len(self.all_dofs_global)
#             cols = matrix.shape[1]
#             # make a grid of indices on interest
#             ixgrid = np.ix_(bcs_to_keep, np.arange(matrix.shape[1]))
#         elif axis == 'column':
#             rows = matrix.shape[0]
#             cols = len(self.all_dofs_global)
#             # make a grid of indices on interest
#             ixgrid = np.ix_(np.arange(matrix.shape[0]),bcs_to_keep)
#         elif axis == 'both':
#             rows = len(self.all_dofs_global)
#             cols = rows
#             # make a grid of indices on interest
#             ixgrid = np.ix_(bcs_to_keep, bcs_to_keep)
#         else:
#             err_msg =  "The extension mode with input \"" + axis
#             err_msg +=  "\" for axis is not avaialbe \n"
#             err_msg += "Choose one of: \"row\", \"column\", \"both\""
#             raise Exception(err_msg)
       
#         extended_matrix = np.zeros((rows, cols))
#         # copy the needed element into the extended matrix
#         extended_matrix[ixgrid] = matrix 

#         return extended_matrix

#     def _get_nodal_coordinates(self, level_height, num_of_levels):
#         # NOTE: adding lower node here as well
#         nodal_coordinates = level_height * np.arange(0, num_of_levels+1)
        
#         # TODO: make consistent
#         # nodal_coordinates = level_height * np.arange(1, num_of_levels+1)
#         return nodal_coordinates

#     def _calculate_mass(self, rho, area, level_height, num_of_levels):
#         """
#         Getting the consistant mass matrix based on analytical integration

#         USING the consistent mass formulation

#         mass values for one level
#         VERSION 3: from Appendix A - Straight Beam Element Matrices - page 228
#         https://link.springer.com/content/pdf/bbm%3A978-3-319-56493-7%2F1.pdf
#         """

#         t = 45.0 # t: beam thickness (y) [m]
#         h = 30.0 # h: beam height (z) [m]
#         rho = 160.0 # density of steel [kg/mˆ3]
#         E = 2.861e8 # E: Young's modulus of steel [N/mˆ2]
#         nu = 3/10 # nu: Poisson's ratio

#         G = E/2/(1+nu) # G: Shear modulus [N/mˆ2]
#         l = level_height # l: beam element length
#         A = t*h # beam area [mˆ2]
#         ASy = 5/6*A    
#         ASz = 5/6*A # effective area of shear
#         Iy = 1/12*h**3*t    
#         Iz = 1/12*t**3*h # second moments of area [mˆ4]
#         Ip = 1/12*t*h*(h**2+t**2) # polar moment of inertia [mˆ4]
#         It = min([h,t])**3 *max([h,t])/7 # torsion constant [mˆ4]
#         Py = 12*E*Iz/(G*ASy*l**2) # relative importance of the shear deformations to the bending deformation
#         Pz = 12*E*Iy/(G*ASz*l**2) # relative importance of the shear deformations to the bending deformation

#         length = level_height
#         m_const = rho * A * length

#         #
#         # mass values for one level
#         # define component-wise to have enable better control for various optimization parameters

#         # axial inertia - along axis x - here marked as x
#         m_x = m_const / 6.0 
#         m_x_11 = 2.
#         m_x_12 = 1.
#         m_el_x = m_x * np.array([[m_x_11, m_x_12],
#                                  [m_x_12, m_x_11]])
#         # torsion inertia - around axis x - here marked as alpha - a
#         m_a = m_const * Ip/A / 6.0 
#         m_a_11 = 2
#         m_a_12 = 1
#         m_el_a = m_a * np.array([[m_a_11, m_a_12],
#                                  [m_a_12, m_a_11]])

#         # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
#         # translation
#         m_yg = m_const / 210 / (1+Py)**2        
#         #
#         m_yg_11 = 70*Py**2 + 147*Py + 78
#         m_yg_12 = (35*Py**2 + 77*Py + 44) * length / 4
#         m_yg_13 = 35*Py**2 + 63*Py + 27
#         m_yg_14 = -(35*Py**2 + 63*Py + 26) * length / 4
#         #
#         m_yg_22 = (7*Py**2 + 14*Py + 8) * length **2 / 4
#         m_yg_23 = - m_yg_14 
#         m_yg_24 = -(7*Py**2 + 14*Py + 6) * length **2 / 4
#         #
#         m_yg_33 = m_yg_11
#         m_yg_34 = -m_yg_12
#         #
#         m_yg_44 = m_yg_22
#         #
#         m_el_yg_trans = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
#                                          [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
#                                          [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
#                                          [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])
#         # rotation
#         m_yg = rho*Iz / 30 / (1+Py)**2 / length        
#         #
#         m_yg_11 = 36
#         m_yg_12 = -(15*Py-3) * length
#         m_yg_13 = -m_yg_11
#         m_yg_14 = m_yg_12
#         #
#         m_yg_22 = (10*Py**2 + 5*Py + 4) * length **2
#         m_yg_23 = - m_yg_12
#         m_yg_24 = (5*Py**2 - 5*Py -1) * length **2
#         #
#         m_yg_33 = m_yg_11
#         m_yg_34 = - m_yg_12
#         #
#         m_yg_44 = m_yg_22
#         #
#         m_el_yg_rot = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
#                                        [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
#                                        [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
#                                        [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

#         # sum up translation and rotation
#         m_el_yg = m_el_yg_trans + m_el_yg_rot

#         # bending - inertia along axis z, rotations around axis y - here marked as beta - b
#         # translation
#         m_zb = m_const / 210 / (1+Pz)**2        
#         #
#         m_zb_11 = 70*Pz**2 + 147*Pz + 78
#         m_zb_12 = -(35*Pz**2 + 77*Pz + 44) * length / 4
#         m_zb_13 = 35*Pz**2 + 63*Pz + 27
#         m_zb_14 = (35*Pz**2 + 63*Pz + 26) * length / 4
#         #
#         m_zb_22 = (7*Pz**2 + 14*Pz + 8) * length **2 / 4
#         m_zb_23 = -m_zb_14
#         m_zb_24 = -(7*Pz**2 + 14*Pz + 6) * length **2 / 4
#         #
#         m_zb_33 = m_zb_11
#         m_zb_34 = - m_zb_12
#         #
#         m_zb_44 = m_zb_22
#         #
#         m_el_zb_trans = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
#                                          [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
#                                          [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
#                                          [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
#         # rotation
#         m_zb = rho*Iy / 30 / (1+Pz)**2 / length        
#         #
#         m_zb_11 = 36
#         m_zb_12 = (15*Pz-3) * length
#         m_zb_13 = -m_zb_11
#         m_zb_14 = m_zb_12
#         #
#         m_zb_22 = (10*Pz**2 + 5*Pz + 4) * length **2
#         m_zb_23 = -m_zb_12
#         m_zb_24 = (5*Pz**2 - 5*Pz -1) * length ** 2
#         #
#         m_zb_33 = m_zb_11
#         m_zb_34 = -m_zb_12
#         #
#         m_zb_44 = m_zb_22
#         #
#         m_el_zb_rot = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
#                                        [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
#                                        [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
#                                        [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
        
#         # sum up translation and rotation
#         m_el_zb = m_el_zb_trans + m_el_zb_rot

#         # assemble all components
#         m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0.,                 m_el_x[0][1], 0., 0., 0., 0., 0.],
#                          [0., m_el_yg[0][0], 0., 0., 0., m_el_yg[0][1],     0., m_el_yg[0][2], 0., 0., 0., m_el_yg[0][3]],
#                          [0., 0., m_el_zb[0][0], 0., m_el_zb[0][1], 0.,     0., 0., m_el_zb[0][2], 0., m_el_zb[0][3], 0.],
#                          [0., 0., 0., m_el_a[0][0], 0., 0.,                 0., 0., 0., m_el_a[0][1], 0., 0.],
#                          [0., 0., m_el_zb[0][1], 0., m_el_zb[1][1], 0.,     0., 0., m_el_zb[1][2], 0., m_el_zb[1][3], 0.],
#                          [0., m_el_yg[0][1], 0., 0., 0., m_el_yg[1][1],     0., m_el_yg[1][2], 0., 0., 0., m_el_yg[1][3]],
                         
#                          [m_el_x[1][0], 0., 0., 0., 0., 0.,                 m_el_x[1][1], 0., 0., 0., 0., 0.],
#                          [0., m_el_yg[0][2], 0., 0., 0., m_el_yg[1][2],     0., m_el_yg[2][2], 0., 0., 0., m_el_yg[2][3]],
#                          [0., 0., m_el_zb[0][2], 0., m_el_zb[1][2], 0.,     0., 0., m_el_zb[2][2], 0., m_el_zb[2][3], 0.],
#                          [0., 0., 0., m_el_a[1][0], 0., 0.,                 0., 0., 0., m_el_a[1][1], 0., 0.],
#                          [0., 0., m_el_zb[0][3], 0., m_el_zb[1][3], 0.,     0., 0., m_el_zb[2][3], 0., m_el_zb[3][3], 0.],
#                          [0., m_el_yg[0][3], 0., 0., 0., m_el_yg[1][3],     0., m_el_yg[2][3], 0., 0., 0., m_el_yg[3][3]]])

#         # global mass matrix initialization with zeros
#         m_glob = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                            (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))


#         # fill global mass matix entries
#         for i in range(num_of_levels):
#             m_temp = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                                (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))

#             # beam part
#             m_temp[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
#                    StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] = m_el

#             m_glob += m_temp

#         # return mass matrix
#         return m_glob


#     def _calculate_stiffness(self, m, level_height, num_of_levels, target_freq, target_mode):
#         """
#         Calculate uniform stiffness k_scalar. A uniform stiffness is assumed for all
#         the elements and the value is calculated using an optimization (or "tuning")
#         for a target frequency of a target mode.
#         """
#         print("Calculating stiffness k in MDoFMixedModel derived class \n")

#         # setup k_scalar_guess as the input for the standard k for a shear-type
#         # MDoF
#         # guess stiffness values for shear and bending
#         # as initial value for optimization

#         # k_scalar_g = 1000.

#         # # using partial to fix some parameters for the
#         # optimizable_function = partial(self._calculate_frequency_error_for_current_k_scalar,
#         #                                m,
#         #                                level_height,
#         #                                num_of_levels,
#         #                                target_freq,
#         #                                target_mode)

#         # print("Optimization for the target k matrix in MDoFMixedModel \n")
#         # minimization_result = minimize(optimizable_function,
#         #                                k_scalar_g, method='Powell',
#         #                                options={'disp': True})

#         # # returning only one value!
#         # k_scalar_opt = minimization_result.x
#         # print("K scalar optimized: ", k_scalar_opt, '\n')

#         return self._assemble_k(level_height, num_of_levels)

#     def _assemble_k(self, level_height, num_of_levels):
#         """
#         Getting the consistant mass matrix based on analytical integration

#         USING the consistent mass formulation

#         stiffness values for one level
#         VERSION 2
        
#         NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
#         seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
#         implemented mass matrices similar to the stiffness one
#         """

#         t = 45.0 # t: beam thickness (y) [m]
#         h = 30.0 # h: beam height (z) [m]
#         rho = 160.0 # density of steel [kg/mˆ3]
#         E = 2.861e8 # E: Young's modulus of steel [N/mˆ2]
#         nu = 3/10 # nu: Poisson's ratio

#         G = E/2/(1+nu) # G: Shear modulus [N/mˆ2]
#         l = level_height # l: beam element length
#         A = t*h # beam area [mˆ2]
#         ASy = 5/6*A    
#         ASz = 5/6*A # effective area of shear
#         Iy = 1/12*h**3*t    
#         Iz = 1/12*t**3*h # second moments of area [mˆ4]
#         Ip = 1/12*t*h*(h**2+t**2) # polar moment of inertia [mˆ4]
#         It = min([h,t])**3 *max([h,t])/7 # torsion constant [mˆ4]
#         Py = 12*E*Iz/(G*ASy*l**2) #
#         Pz = 12*E*Iy/(G*ASz*l**2) #

#         length = l
#         # axial stiffness - along axis x - here marked as x
#         k_x = E*A/l
#         k_x_11 = 1.0
#         k_x_12 = -1.0
#         k_el_x = k_x * np.array([[k_x_11, k_x_12],
#                                  [k_x_12, k_x_11]])
#         # torsion stiffness - around axis x - here marked as alpha - a
#         k_a = G*It/l  # G*K/l
#         k_a_11 = 1.0
#         k_a_12 = -1.0
#         k_el_a = k_a * np.array([[k_a_11, k_a_12],
#                                  [k_a_12, k_a_11]])
#         # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
#         beta_yg = Py
#         k_yg = E*Iz/(1+beta_yg)/l**3
#         #
#         k_yg_11 = 12.
#         k_yg_12 = 6. * length
#         k_yg_13 = -k_yg_11
#         k_yg_14 = k_yg_12
#         #
#         k_yg_22 = (4.+beta_yg) * length **2
#         k_yg_23 = -k_yg_12
#         k_yg_24 = (2-beta_yg) * length ** 2
#         #
#         k_yg_33 = k_yg_11
#         k_yg_34 = -k_yg_12
#         #
#         k_yg_44 = k_yg_22
#         #
#         k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
#                                    [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
#                                    [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
#                                    [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

#         # bending - displacement along axis z, rotations around axis y - here marked as beta - b
#         beta_zb = Pz
#         k_zb = E*Iy/(1+beta_zb)/l**3
#         #
#         k_zb_11 = 12.
#         k_zb_12 = -6. * length
#         k_zb_13 = -12.
#         k_zb_14 = k_zb_12
#         #
#         k_zb_22 = (4.+beta_zb) * length **2
#         k_zb_23 = -k_zb_12 
#         k_zb_24 = (2-beta_zb) * length ** 2
#         #
#         k_zb_33 = k_zb_11
#         k_zb_34 = - k_zb_12
#         #
#         k_zb_44 = k_zb_22
#         #
#         k_el_zb = k_zb * np.array([[k_zb_11, k_zb_12, k_zb_13, k_zb_14],
#                                    [k_zb_12, k_zb_22, k_zb_23, k_zb_24],
#                                    [k_zb_13, k_zb_23, k_zb_33, k_zb_34],
#                                    [k_zb_14, k_zb_24, k_zb_34, k_zb_44]])

#         # assemble all components
#         k_el = np.array([[k_el_x[0][0], 0., 0., 0., 0., 0.,                 k_el_x[0][1], 0., 0., 0., 0., 0.],
#                          [0., k_el_yg[0][0], 0., 0., 0., k_el_yg[0][1],     0., k_el_yg[0][2], 0., 0., 0., k_el_yg[0][3]],
#                          [0., 0., k_el_zb[0][0], 0., k_el_zb[0][1], 0.,     0., 0., k_el_zb[0][2], 0., k_el_zb[0][3], 0.],
#                          [0., 0., 0., k_el_a[0][0], 0., 0.,                 0., 0., 0., k_el_a[0][1], 0., 0.],
#                          [0., 0., k_el_zb[0][1], 0., k_el_zb[1][1], 0.,     0., 0., k_el_zb[1][2], 0., k_el_zb[1][3], 0.],
#                          [0., k_el_yg[0][1], 0., 0., 0., k_el_yg[1][1],     0., k_el_yg[1][2], 0., 0., 0., k_el_yg[1][3]],
                         
#                          [k_el_x[1][0], 0., 0., 0., 0., 0.,                 k_el_x[1][1], 0., 0., 0., 0., 0.],
#                          [0., k_el_yg[0][2], 0., 0., 0., k_el_yg[1][2],     0., k_el_yg[2][2], 0., 0., 0., k_el_yg[2][3]],
#                          [0., 0., k_el_zb[0][2], 0., k_el_zb[1][2], 0.,     0., 0., k_el_zb[2][2], 0., k_el_zb[2][3], 0.],
#                          [0., 0., 0., k_el_a[1][0], 0., 0.,                 0., 0., 0., k_el_a[1][1], 0., 0.],
#                          [0., 0., k_el_zb[0][3], 0., k_el_zb[1][3], 0.,     0., 0., k_el_zb[2][3], 0., k_el_zb[3][3], 0.],
#                          [0., k_el_yg[0][3], 0., 0., 0., k_el_yg[1][3],     0., k_el_yg[2][3], 0., 0., 0., k_el_yg[3][3]]])

#         # global stiffness matrix initialization with zeros
#         k_glob = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                            (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))


#         # fill global stiffness matix entries
#         for i in range(num_of_levels):
#             k_temp = np.zeros(((num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size],
#                                (num_of_levels+1) * StraightBeam.DOFS_PER_NODE[self.domain_size]))

#             # beam part
#             k_temp[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
#                    StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] = k_el

#             k_glob += k_temp

#         # return stiffness matrix
#         return k_glob

#     def _calculate_damping(self, m, k, zeta):
#         """
#         Calculate damping b based upon the Rayleigh assumption
#         using the first 2 eigemodes - here generically i and i
#         """
#         print("Calculating damping b in MDoFMixedModel derived class \n")
#         mode_i = 0
#         mode_j = 1
#         zeta_i = zeta
#         zeta_j = zeta

#         # TODO: try to avoid this code duplication
#         # raw results

#         # apply BC's to be able to solve and avoid rigid modes

#         eig_values_raw, eigen_modes_raw = linalg.eigh(self.apply_bc_by_reduction(k), self.apply_bc_by_reduction(m))
#         # rad/s
#         eig_values = np.sqrt(np.real(eig_values_raw))
#         # 1/s = Hz
#         eig_freqs = eig_values / 2. / np.pi
#         # sort eigenfrequencies
#         eig_freqs_sorted_indices = np.argsort(eig_freqs)
#         #

#         a = np.linalg.solve(0.5 *
#                             np.array(
#                                 [[1 / eig_values[eig_freqs_sorted_indices[mode_i]],
#                                   eig_values[
#                                   eig_freqs_sorted_indices[mode_i]]],
#                                     [1 / eig_values[eig_freqs_sorted_indices[mode_j]],
#                                      eig_values[
#                                      eig_freqs_sorted_indices[
#                                          mode_j]]]]),
#                             [zeta_i, zeta_j])
#         return a[0] * m + a[1] * k

#     # k_scalar, k_beam):
#     def _calculate_frequency_error_for_current_k_scalar(self, m, level_height, num_of_levels, target_freq, target_mode, k_scalar):
#         k = self._assemble_k(level_height, num_of_levels)

#         # TODO: try to avoid this code duplication
#         # raw results
#         eig_values_raw, eigen_modes_raw = linalg.eigh(k, m)
#         # rad/s
#         eig_values = np.sqrt(np.real(eig_values_raw))
#         # 1/s = Hz
#         eig_freqs = eig_values / 2. / np.pi
#         # sort eigenfrequencies
#         eig_freqs_sorted_indices = np.argsort(eig_freqs)
#         #

#         current_target_freq = eig_freqs[eig_freqs_sorted_indices[target_mode-1]]

#         return (target_freq - current_target_freq) ** 2 / target_freq**2

#     def _load_distribution(self, level_height, num_of_levels, wind_forces):

#         # Nodal distribution the applis load on a beam model
#         def nodal_force(force):
#             F = [force / 2, force * level_height / 12,
#                  force / 2, -force * level_height / 12]
#             return F

#         nodal_load = list(map(nodal_force, wind_forces))

#         load = np.zeros(2 * num_of_levels + 2)

#         for i in range(num_of_levels):
#             load_temp = np.zeros(2 * num_of_levels + 2)
#             load_temp[2 * i:2 * i + 4] = nodal_load[i]
#             load += load_temp

#         # remove the fixed degrees of freedom
#         rdof = [1, 0]
#         for dof in rdof:
#             load = np.delete(load, dof, axis=0)

#         return load
