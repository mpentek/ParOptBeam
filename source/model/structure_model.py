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

from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
from source.auxiliary.auxiliary_functionalities import evaluate_polynomial


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

    THRESHOLD = 1e-8

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "domain_size": "3D",
        "system_parameters": {},
        "boundary_conditions": "fixed-free",
        "elastic_fixity_dofs": {}}

    def __init__(self, parameters):

        # validating and assign model parameters
        validate_and_assign_defaults(StraightBeam.DEFAULT_SETTINGS, parameters)

        # TODO: add domain size check
        self.domain_size = parameters["domain_size"]

        # TODO: validate and assign parameters
        # NOTE: for now using the assumption of the prismatic homogenous isotropic beam
        self.parameters = {
            # material
            'rho': parameters["system_parameters"]["material"]["density"],
            'e': parameters["system_parameters"]["material"]["youngs_modulus"],
            'nu': parameters["system_parameters"]["material"]["poisson_ratio"],
            'zeta': parameters["system_parameters"]["material"]["damping_ratio"],
            # geometric
            'lx': parameters["system_parameters"]["geometry"]["length_x"],
            'n_el': parameters["system_parameters"]["geometry"]["number_of_elements"]} 
        
        # defined on intervals as piecewise continous function on an interval starting from 0.0
        self.parameters['intervals'] = []
        for val in parameters["system_parameters"]["geometry"]["defined_on_intervals"]:
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
        self.initialize_user_defined_geometric_parameters()
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

        # initialize empty place holders for point stiffness and mass entries
        self.point_stiffness = {'idxs': [], 'vals': []}
        self.point_mass = {'idxs': [], 'vals': []}

        self.parameters["boundary_conditions"] = parameters["boundary_conditions"]
        self.apply_bcs()

        # after initial setup
        self.calculate_global_matrices()

    def apply_elastic_bcs(self):
        # handle potential elastic BCs
        self.elastic_bc_dofs = {}
        if 'elastic_fixity_dofs' in self.parameters:
            elastic_bc_dofs_tmp = self.parameters["elastic_fixity_dofs"]
        else:
            print(
                'parameters does not have "elastic_fixity_dofs"')
            elastic_bc_dofs_tmp = {}
        
        for key in elastic_bc_dofs_tmp:
            # TODO: check if type cast to int is robust enough
            if int(key) not in self.bc_dofs:
                err_msg = "The elastic BC dof for input \"" + key
                err_msg += "\" is not available for " + \
                    self.parameters["boundary_conditions"] + "\n"
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

                # now inserting the elastic dofs into the generic formulation
                # affects only diagonal entries
                self.point_stiffness['idxs'].append([val,val])
                # with this additional value
                self.point_stiffness['vals'].append(elastic_bc_dofs_tmp[key])

    def apply_bcs(self):
        # TODO: make BC handling cleaner and compact
        self.all_dofs_global = np.arange(
            self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size])
        bc = '\"' + \
            self.parameters["boundary_conditions"] + '\"'
        if bc in StraightBeam.AVAILABLE_BCS:
            # NOTE: create a copy of the list - useful if some parametric study is done
            self.bc_dofs = StraightBeam.BC_DOFS[self.domain_size][bc][:]
        else:
            err_msg = "The BC for input \"" + \
                self.parameters["boundary_conditions"]
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join(StraightBeam.AVAILABLE_BCS)
            raise Exception(err_msg)

        self.apply_elastic_bcs()
        
        # list copy by slicing -> [:] -> to have a copy by value
        bc_dofs_global = self.bc_dofs[:]
        for idx, dof in enumerate(bc_dofs_global):
            # shift to global numbering the negative values
            if dof < 0:
                bc_dofs_global[idx] = dof + len(self.all_dofs_global)

        # only take bc's of interest
        self.dofs_to_keep = list(set(self.all_dofs_global)-set(bc_dofs_global)) 

    def initialize_user_defined_geometric_parameters(self):
        # geometric
        # characteristics lengths
        self.parameters['ly'] = [self.evaluate_characteristic_on_interval(
            x, 'c_ly') for x in self.parameters['x']]
        self.parameters['lz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_lz') for x in self.parameters['x']]
        self.charact_length = (np.mean(self.parameters['ly']) + np.mean(self.parameters['lz']))/2

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

    def calculate_global_matrices(self):
        # mass matrix
        self.m = self._get_mass()
        # stiffness matrix
        self.k = self._get_stiffness()
        # damping matrix - needs to be done after mass and stiffness as Rayleigh method nees these
        self.b = self._get_damping()


    def decompose_and_quantify_eigenmodes(self, considered_modes=10):
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.dofs_to_keep)
        else:
            if considered_modes > len(self.dofs_to_keep):
                considered_modes = len(self.dofs_to_keep)
        
        self.eigenvalue_solve()
        
        self.decomposed_eigenmodes = {'values': [], 'rel_contribution' : []}

        for i in range(considered_modes):
            decomposed_eigenmode = {}
            rel_contrib = {}
            selected_mode = self.eig_freqs_sorted_indices[i]

            for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.domain_size])),
                                StraightBeam.DOF_LABELS[self.domain_size]):
                start = idx
                step = StraightBeam.DOFS_PER_NODE[self.domain_size]
                stop = self.eigen_modes_raw.shape[0] + idx - step
                decomposed_eigenmode[label] = self.eigen_modes_raw[start:stop +
                                                                            1:step][:, selected_mode]
                if label in ['a', 'b', 'g']:
                    # for rotation dofs multiply with a characteristic length
                    # to make comparable to translation dofs
                    rel_contrib[label] = self.charact_length * linalg.norm(decomposed_eigenmode[label])
                else:
                    # for translatio dofs
                    rel_contrib[label] = linalg.norm(decomposed_eigenmode[label])

            self.decomposed_eigenmodes['values'].append(decomposed_eigenmode)
            self.decomposed_eigenmodes['rel_contribution'].append(rel_contrib)


    def identify_decoupled_eigenmodes(self, considered_modes=10, print_to_console=False):
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.dofs_to_keep)
        else:
            if considered_modes > len(self.dofs_to_keep):
                considered_modes = len(self.dofs_to_keep)
        

        self.decompose_and_quantify_eigenmodes()

        self.mode_identification_results = {}

        for i in range(considered_modes):

            selected_mode = self.eig_freqs_sorted_indices[i]

            for case_id in StraightBeam.MODE_CATEGORIZATION[self.domain_size]:
                match_for_case_id = False

                for dof_contribution_id in StraightBeam.MODE_CATEGORIZATION[self.domain_size][case_id]:
                    if self.decomposed_eigenmodes['rel_contribution'][i][dof_contribution_id] > StraightBeam.THRESHOLD:
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
            ixgrid = np.ix_(self.dofs_to_keep, np.arange(matrix.shape[1]))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(self.all_dofs_global)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.dofs_to_keep)
        elif axis == 'both':
            rows = len(self.all_dofs_global)
            cols = rows
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)
        elif axis == 'row_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.dofs_to_keep, [0])
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
        #ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)

        # create new array with zeros the size it should be
        # with ixgrid take from existing the relevant data and copy to new
        if axis == 'row':
            rows = len(self.all_dofs_global)
            cols = matrix.shape[1]
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, np.arange(matrix.shape[1]))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(self.all_dofs_global)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.dofs_to_keep)
        elif axis == 'both':
            rows = len(self.all_dofs_global)
            cols = rows
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)
        elif axis == 'row_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.dofs_to_keep, [0])
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

    # NOTE: not used forn now
    def _assemble_el_into_glob(self, el_matrix):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        return glob_matrix

    def __get_el_mass(self, i):
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

        if self.domain_size == '3D':
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

        if self.domain_size == '3D':    
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
        if self.domain_size == '3D':
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

        elif self.domain_size == '2D':
            m_el = np.array([[m_el_x[0][0], 0., 0.,                 m_el_x[0][1], 0., 0.],
                             [0., m_el_yg[0][0], m_el_yg[0][1],     0., m_el_yg[0][2], m_el_yg[0][3]],
                             [0., m_el_yg[0][1], m_el_yg[1][1],     0., m_el_yg[1][2], m_el_yg[1][3]],
                             
                             [m_el_x[1][0], 0., 0.,                 m_el_x[1][1], 0., 0.],
                             [0., m_el_yg[0][2], m_el_yg[1][2],     0., m_el_yg[2][2], m_el_yg[2][3]],
                             [0., m_el_yg[0][3], m_el_yg[1][3],     0., m_el_yg[2][3], m_el_yg[3][3]]])

        return m_el

    def _get_mass(self):

        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.__get_el_mass(i)
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        
        for idx, val in zip(self.point_mass['idxs'], self.point_mass['vals']):
            glob_matrix[idx[0],idx[1]] = glob_matrix[idx[0],idx[1]] + val

        return glob_matrix

    def __get_el_stiffness(self, i):
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
        
        if self.domain_size == '3D':
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

        if self.domain_size == '3D':
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

        if self.domain_size == '3D':
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
        
        elif self.domain_size == '2D':
            k_el = np.array([[k_el_x[0][0], 0., 0.,                 k_el_x[0][1], 0., 0.],
                             [0., k_el_yg[0][0], k_el_yg[0][1],     0., k_el_yg[0][2], k_el_yg[0][3]],       
                             [0., k_el_yg[0][1], k_el_yg[1][1],     0., k_el_yg[1][2], k_el_yg[1][3]],

                             [k_el_x[1][0], 0., 0.,                 k_el_x[1][1], 0., 0.],
                             [0., k_el_yg[0][2], k_el_yg[1][2],     0., k_el_yg[2][2], k_el_yg[2][3]],
                             [0., k_el_yg[0][3], k_el_yg[1][3],     0., k_el_yg[2][3], k_el_yg[3][3]]])

        return k_el

    def _get_stiffness(self):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * StraightBeam.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.__get_el_stiffness(i)
            glob_matrix[StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL,
                        StraightBeam.DOFS_PER_NODE[self.domain_size] * i: StraightBeam.DOFS_PER_NODE[self.domain_size] * i + StraightBeam.DOFS_PER_NODE[self.domain_size] * StraightBeam.NODES_PER_LEVEL] += el_matrix
        
        for idx, val in zip(self.point_stiffness['idxs'], self.point_stiffness['vals']):
            glob_matrix[idx[0],idx[1]] = glob_matrix[idx[0],idx[1]] + val

        return glob_matrix

    def _get_damping(self):
        """
        Calculate damping b based upon the Rayleigh assumption
        using the first 2 eigemodes - here generically i and i
        """

        mode_i = 0
        mode_j = 1
        zeta_i = self.parameters['zeta']
        zeta_j = zeta_i

        self.eigenvalue_solve()

        self.a = np.linalg.solve(0.5 *
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
        return self.a[0] * self.m + self.a[1] * self.k