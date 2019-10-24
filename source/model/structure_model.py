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

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from source.auxiliary.auxiliary_functionalities import evaluate_polynomial
from source.auxiliary.global_definitions import *
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
from source.element.TimoshenkoBeamElement import TimoshenkoBeamElement


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
        6. test unit how many elements it works, also specific cases with analystical solutions
    """

    def __init__(self, parameters):
        # TODO: add number of considered modes for output parameters upper level
        # also check redundancy with eigenvalue analysis

        # validating and assign model parameters
        validate_and_assign_defaults(DEFAULT_SETTINGS, parameters)

        # TODO: add domain size check
        self.domain_size = parameters["domain_size"]

        # needed to identify output results
        self.name = parameters["name"]

        # TODO: validate and assign parameters
        # NOTE: for now using the assumption of the prismatic homogenous isotropic beam
        self.parameters = {'rho': parameters["system_parameters"]["material"]["density"],
                           'e': parameters["system_parameters"]["material"]["youngs_modulus"],
                           'nu': parameters["system_parameters"]["material"]["poisson_ratio"],
                           'zeta': parameters["system_parameters"]["material"]["damping_ratio"],
                           'lx': parameters["system_parameters"]["geometry"]["length_x"],
                           'n_el': parameters["system_parameters"]["geometry"]["number_of_elements"], 'intervals': []}

        # defined on intervals as piecewise continuous function on an interval starting from 0.0
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
                'c_it': val["torsional_moment_of_inertia"],
                'c_m': val["outrigger_mass"],
                'c_k': val["outrigger_stiffness"]
            })

        # TODO: later probably move to an initialize function
        # material
        # shear modulus
        self.parameters['g'] = self.parameters['e'] / \
                               2 / (1 + self.parameters['nu'])

        # running coordinate x - in the middle of each beam element
        self.parameters['x'] = [(x + 0.5) / self.parameters['n_el'] * self.parameters['lx']
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
                        np.arange(self.parameters['n_el'] + 1)

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

        # define element type
        self.element = TimoshenkoBeamElement(self.parameters, self.domain_size)

        self.n_nodes = self.parameters['n_el'] + 1

        # initialize empty place holders for point stiffness and mass entries
        self.point_stiffness = {'idxs': [], 'vals': []}
        self.point_mass = {'idxs': [], 'vals': []}

        self.parameters["boundary_conditions"] = parameters["boundary_conditions"]
        self.apply_bcs()

        # after initial setup
        self.calculate_global_matrices()
        self.identify_decoupled_eigenmodes()

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
                self.point_stiffness['idxs'].append([val, val])
                # with this additional value
                self.point_stiffness['vals'].append(elastic_bc_dofs_tmp[key])
        # apply the point masses and point stiffness from outtriggers         
        self.apply_point_values()

    def apply_bcs(self):
        # TODO: make BC handling cleaner and compact
        self.all_dofs_global = np.arange(
            self.n_nodes * DOFS_PER_NODE[self.domain_size])
        bc = '\"' + \
             self.parameters["boundary_conditions"] + '\"'
        if bc in AVAILABLE_BCS:
            # NOTE: create a copy of the list - useful if some parametric study is done
            self.bc_dofs = BC_DOFS[self.domain_size][bc][:]
        else:
            err_msg = "The BC for input \"" + \
                      self.parameters["boundary_conditions"]
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join(AVAILABLE_BCS)
            raise Exception(err_msg)

        self.apply_elastic_bcs()

        # list copy by slicing -> [:] -> to have a copy by value
        bc_dofs_global = self.bc_dofs[:]
        for idx, dof in enumerate(bc_dofs_global):
            # shift to global numbering the negative values
            if dof < 0:
                bc_dofs_global[idx] = dof + len(self.all_dofs_global)

        # only take bc's of interest
        self.dofs_to_keep = list(set(self.all_dofs_global) - set(bc_dofs_global))

    def apply_point_values(self):
        # point stiffness and point masses at respective dof for the outtrigger 
        for values in self.parameters['intervals']:
            if values['bounds'][1] == "End":
                outtriger_id = int(self.parameters['lx'] / self.parameters['lx_i'])
            else:
                outtriger_id = int(values['bounds'][1] / self.parameters['lx_i'])
            id_val = outtriger_id * DOFS_PER_NODE[self.domain_size]
            # affects only diagonal entries and for transilation al DOF and not rotational DOF
            for i in range(int(np.ceil(DOFS_PER_NODE[self.domain_size] / 2))):
                self.point_stiffness['idxs'].append([id_val + i, id_val + i])
                self.point_stiffness['vals'].append(values['c_k'])
                self.point_mass['idxs'].append([id_val + i, id_val + i])
                self.point_mass['vals'].append(values['c_m'])
            # out tigger stiffnss in x, Y, and Z the same ?? AK: need to check 

    def initialize_user_defined_geometric_parameters(self):
        # geometric
        # characteristics lengths
        self.parameters['ly'] = [self.evaluate_characteristic_on_interval(
            x, 'c_ly') for x in self.parameters['x']]
        self.parameters['lz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_lz') for x in self.parameters['x']]
        self.charact_length = (np.mean(self.parameters['ly']) + np.mean(self.parameters['lz'])) / 2

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
                self.parameters['g'] * b * self.parameters['lx_i'] ** 2) for a, b in
                                 zip(self.parameters['iz'], self.parameters['a_sy'])]
        self.parameters['pz'] = [12 * self.parameters['e'] * a / (
                self.parameters['g'] * b * self.parameters['lx_i'] ** 2) for a, b in
                                 zip(self.parameters['iy'], self.parameters['a_sz'])]

        if is_bernoulli:
            # NOTE: Bernoulli beam set to 0.0
            self.parameters['py'] = [0.0 for a, b in zip(self.parameters['iz'], self.parameters['a_sy'])]
            self.parameters['pz'] = [0.0 for a, b in zip(self.parameters['iy'], self.parameters['a_sz'])]

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
        # TODO: Add outtrigger masses to this entry
        if print_to_console:
            print('CURRENT:')
            print('total mass ', self.parameters['m_tot'])
            print('density: ', self.parameters['rho'])
            print()

    def calculate_global_matrices(self):
        # using computational values for m,b,k as this reduction is done otherwise many times

        # mass matrix
        self.m = self._get_mass()
        self.comp_m = self.apply_bc_by_reduction(self.m)
        # stiffness matrix
        self.k = self._get_stiffness()
        self.comp_k = self.apply_bc_by_reduction(self.k)
        # damping matrix - needs to be done after mass and stiffness as Rayleigh method nees these
        self.b = self._get_damping()
        self.comp_b = self.apply_bc_by_reduction(self.b)

    def decompose_and_quantify_eigenmodes(self, considered_modes=15):
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.dofs_to_keep)
        else:
            if considered_modes > len(self.dofs_to_keep):
                considered_modes = len(self.dofs_to_keep)

        self.eigenvalue_solve()

        self.decomposed_eigenmodes = {'values': [], 'rel_contribution': [], 'eff_modal_mass': [], 'rel_participation' : []}

        for i in range(considered_modes):
            decomposed_eigenmode = {}
            rel_contrib = {}
            eff_modal_mass = {}
            rel_participation = {}
            selected_mode = self.eig_freqs_sorted_indices[i]

            for idx, label in zip(list(range(DOFS_PER_NODE[self.domain_size])),
                                  DOF_LABELS[self.domain_size]):
                start = idx
                step = DOFS_PER_NODE[self.domain_size]
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

                # adding computation of modal mass
                # TODO: for now using element mass (as constant) and nodal dof value - make consistent
                # IMPORTANT
                if label in ['x', 'y', 'z', 'a']:
                    if rel_contrib[label] > THRESHOLD:
                        eff_modal_numerator = 0.0
                        eff_modal_denominator = 0.0
                        total_mass = 0.0

                        for i in range(len(self.parameters['x'])):
                            storey_mass = self.parameters['a'][i] * \
                                                self.parameters['rho'] * self.parameters['lx_i']
                            if label == 'a':
                                # NOTE for torsion using the equivalency of a rectangle with sides ly_i, lz_i
                                storey_mass *= (self.parameters['lz'][i]**2 + self.parameters['ly'][i]**2)/12

                                # TODO check as torsion 4-5-6 does not seem to be ok in the results

                            total_mass += storey_mass
                            eff_modal_numerator += storey_mass * decomposed_eigenmode[label][i]
                            eff_modal_denominator += storey_mass * decomposed_eigenmode[label][i]**2

                        eff_modal_mass[label] = eff_modal_numerator**2 / eff_modal_denominator
                        rel_participation[label] = eff_modal_mass[label] / total_mass     

                    else:
                        eff_modal_mass[label] = 0.0
                        rel_participation[label] = 0.0
                else:
                    # TODO for now for rotations
                    eff_modal_mass[label] = 0.0
                    rel_participation[label] = 0.0

            self.decomposed_eigenmodes['values'].append(decomposed_eigenmode)
            self.decomposed_eigenmodes['rel_contribution'].append(rel_contrib)
            self.decomposed_eigenmodes['eff_modal_mass'].append(eff_modal_mass)
            self.decomposed_eigenmodes['rel_participation'].append(rel_participation)

    def identify_decoupled_eigenmodes(self, considered_modes=15, print_to_console=False):
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

            for case_id in MODE_CATEGORIZATION[self.domain_size]:
                match_for_case_id = False

                for dof_contribution_id in MODE_CATEGORIZATION[self.domain_size][case_id]:
                    if self.decomposed_eigenmodes['rel_contribution'][i][dof_contribution_id] > THRESHOLD:
                        match_for_case_id = True

                # TODO: check if robust enough for modes where 2 DoFs are involved
                if match_for_case_id:
                    if case_id in self.mode_identification_results:
                        self.mode_identification_results[case_id].append({
                            (selected_mode + 1): [max(self.decomposed_eigenmodes['eff_modal_mass'][i].values()),
                                                  max(self.decomposed_eigenmodes['rel_participation'][i].values())]
                            })
                    else:
                        self.mode_identification_results[case_id] = [{
                            (selected_mode + 1): [max(self.decomposed_eigenmodes['eff_modal_mass'][i].values()),
                                                  max(self.decomposed_eigenmodes['rel_participation'][i].values())]
                            }]

        if print_to_console:
            print('Result of decoupled eigenmode identification for the first ' +
                  str(considered_modes) + ' mode(s)')

            for mode, mode_ids in self.mode_identification_results.items():
                print('  Mode:', mode)
                for mode_id in mode_ids:
                    m_id = list(mode_id.keys())[0]
                    # TODO use different datatype to avoid list(mode_id.keys())[0]
                    print('    Eigenform ' + str(m_id) + ' with eigenfrequency ' + '{:.2f}'.format(
                        self.eig_freqs[self.eig_freqs_sorted_indices[m_id - 1]]) + ' Hz')

    def eigenvalue_solve(self):
        # raw results
        # solving for reduced m and k - applying BCs leads to avoiding rigid body modes
        self.eig_values_raw, self.eigen_modes_raw = linalg.eigh(
            self.comp_k, self.comp_m)
        # rad/s
        self.eig_values = np.sqrt(np.real(self.eig_values_raw))
        self.eig_freqs = self.eig_values / 2. / np.pi
        self.eig_pers = 1 / self.eig_freqs
        # sort eigenfrequencies

        # NOTE: it seems that it is anyway sorted
        # TODO: check if it can at all happen that it is not sorted, otherwise operation superflous
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
                    return evaluate_polynomial(running_coord - val['bounds'][0], val[characteristic_identifier])
            elif "End" in val['bounds']:
                if val['bounds'][0] <= running_coord and running_coord <= self.parameters['lx']:
                    return evaluate_polynomial(running_coord - val['bounds'][0], val[characteristic_identifier])

    def plot_model_properties(self, pdf_report, display_plot, print_to_console=False):

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

        if pdf_report is not None:
            pdf_report.savefig()
            plt.close(fig)

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

        if pdf_report is not None:
            pdf_report.savefig()
            plt.close(fig)

        fig = plt.figure(3)
        plt.plot(self.parameters['x'], self.parameters['ly'],
                 'r-', marker='*', label='ly')
        plt.plot(self.parameters['x'], self.parameters['lz'],
                 'g-', marker='^', label='lz')
        plt.legend()
        plt.grid()

        if pdf_report is not None:
            pdf_report.savefig()
            plt.close(fig)

        fig = plt.figure(4)
        plt.plot(self.parameters['x'], self.parameters['py'],
                 'r-', marker='*', label='py')
        plt.plot(self.parameters['x'], self.parameters['pz'],
                 'g-', marker='^', label='pz')
        plt.legend()
        plt.grid()

        if pdf_report is not None:
            pdf_report.savefig()
            plt.close(fig)

        if display_plot:
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
        # ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)

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

    # NOTE: not used for now
    def _assemble_el_into_glob(self, el_matrix):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matrix entries
        for i in range(self.parameters['n_el']):
            glob_matrix[
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[
                                                     self.domain_size] * NODES_PER_LEVEL,
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[
                                                     self.domain_size] * NODES_PER_LEVEL] += el_matrix
        return glob_matrix

    def _get_mass(self):

        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.element.get_el_mass(i)
            glob_matrix[
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[
                                                     self.domain_size] * NODES_PER_LEVEL,
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[
                                                     self.domain_size] * NODES_PER_LEVEL] += el_matrix

        for idx, val in zip(self.point_mass['idxs'], self.point_mass['vals']):
            glob_matrix[idx[0], idx[1]] = glob_matrix[idx[0], idx[1]] + val

        return glob_matrix

    def _get_stiffness(self):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matix entries
        for i in range(self.parameters['n_el']):
            el_matrix = self.element.get_el_stiffness(i)
            glob_matrix[
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[self.domain_size] * NODES_PER_LEVEL,
            DOFS_PER_NODE[self.domain_size] * i: DOFS_PER_NODE[self.domain_size] * i +
                                                 DOFS_PER_NODE[self.domain_size] * NODES_PER_LEVEL] += el_matrix

        for idx, val in zip(self.point_stiffness['idxs'], self.point_stiffness['vals']):
            glob_matrix[idx[0], idx[1]] = glob_matrix[idx[0], idx[1]] + val

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
