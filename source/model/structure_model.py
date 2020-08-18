import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from os.path import join as os_join

from source.auxiliary.auxiliary_functionalities import evaluate_polynomial
import source.auxiliary.global_definitions as GD
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.writer_utilitites as writer_utilities


class StraightBeam(object):
    """
    A 2D/3D prismatic homogeneous isotropic Timoshenko beam element
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

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "name": "this_model_name",
        "domain_size": "3D",
        "system_parameters": {},
        "boundary_conditions": "fixed-free",
        "elastic_fixity_dofs": {}}

    def __init__(self, parameters):
        # TODO: add number of considered modes for output parameters upper level
        # also check redundancy with eigenvalue analysis

        # validating and assign model parameters
        self.bc_element_dofs = []
        self.bc_dofs = []
        validate_and_assign_defaults(self.DEFAULT_SETTINGS, parameters)

        # TODO: add domain size check
        self.domain_size = parameters["domain_size"]

        # needed to identify output results
        self.name = parameters["name"]

        # TODO: validate and assign parameters
        # NOTE: for now using the assumption of the prismatic homogeneous isotropic beam
        self.parameters = {'rho': parameters["system_parameters"]["material"]["density"],
                           'e': parameters["system_parameters"]["material"]["youngs_modulus"],
                           'nu': parameters["system_parameters"]["material"]["poisson_ratio"],
                           'zeta': parameters["system_parameters"]["material"]["damping_ratio"],
                           'lx': parameters["system_parameters"]["geometry"]["length_x"],
                           'n_el': parameters["system_parameters"]["geometry"]["number_of_elements"],
                           'element_type': parameters["system_parameters"]["element_params"]["type"],
                           'is_nonlinear': parameters["system_parameters"]["element_params"]["is_nonlinear"],
                           'intervals': []}

        # defined on intervals as piecewise continuous function on an interval starting from 0.0
        for idx, val in enumerate(parameters["system_parameters"]["geometry"]["defined_on_intervals"]):
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
            try: 
                self.parameters["intervals"][idx]['m'] = val["outrigger"]["mass"]
                self.parameters["intervals"][idx]['out_stif_y'] = val["outrigger"]["stiffness_ratio_y"]
                self.parameters["intervals"][idx]['out_stif_z'] = val["outrigger"]["stiffness_ratio_z"]
            except:
                self.parameters["intervals"][idx]['m'] = None
                self.parameters["intervals"][idx]['out_stif_y'] = None
                self.parameters["intervals"][idx]['out_stif_z'] = None
                print('No outrigger mass for interval ' + str(idx))

        # define element type
        self.n_elements = self.parameters['n_el']
        self.n_nodes = self.n_elements + 1
        # placeholder for solutions
        self.lx_i = self.parameters['lx'] / self.parameters['n_el']
        self.nodal_coordinates = {}
        self.elements = []
        self.initialize_elements()

        # initialize empty place holders for point stiffness and mass entries
        # elastic bcs and outriggers might contribute to these
        # using dict syntax -> for one dof entry onle one value is permitted
        self.point_stiffness = {}
        self.point_mass = {}

        # matrices
        self.m = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                           self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))
        self.comp_m = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))
        self.k = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                           self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))
        self.comp_k = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))
        self.b = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                           self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))
        self.comp_b = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))

        # boundary conditions
        self.all_dofs_global = np.arange(
            self.n_nodes * GD.DOFS_PER_NODE[self.domain_size])

        self.elastic_bc_dofs = {}
        self.parameters["boundary_conditions"] = parameters["boundary_conditions"]

        # internally calls apply_elastic_bcs() which might contribute to point_values
        self.apply_bcs()

        self.update_equivalent_nodal_mass()
        # might contribute to point_values
        self.update_outrigger_contribution()
        # compute total mass
        self.calculate_total_mass()

        # after initial setup
        self.calculate_global_matrices()
        self.calculate_total_mass()

        self.mode_identification_results = {}
        self.decomposed_eigenmodes = {'values': [], 'rel_contribution': []}
        self.identify_decoupled_eigenmodes()

    def initialize_elements(self):
        # running coordinate x - in the middle of each beam element
        self.parameters['x'] = [x / self.n_elements * self.parameters['lx']
                                for x in list(range(self.n_nodes))]
        self.parameters['x_mid'] = [(x + 0.5) / self.n_elements * self.parameters['lx']
                                    for x in list(range(self.n_elements))]

        # geometric
        self.parameters['ly'] = [self.evaluate_characteristic_on_interval(
            x, 'c_ly') for x in self.parameters['x']]
        self.parameters['lz'] = [self.evaluate_characteristic_on_interval(
            x, 'c_lz') for x in self.parameters['x']]
        # characteristics lengths
        self.charact_length = (
                                      np.mean(self.parameters['ly']) + np.mean(self.parameters['lz'])) / 2

        for i in range(self.n_elements):
            element_params = self.initialize_element_geometric_parameters(i)
            coord = np.array([[self.parameters['x'][i], 0.0, 0.0],
                              [self.parameters['x'][i + 1], 0.0, 0.0]])
            if self.parameters['element_type'] == "Timoshenko":
                from source.element.timoshenko_beam_element import TimoshenkoBeamElement
                e = TimoshenkoBeamElement(
                    self.parameters, element_params, coord, i, self.domain_size)
            elif self.parameters['element_type'] == "Bernoulli":
                from source.element.bernouli_beam_element import BernoulliBeamElement
                e = BernoulliBeamElement(
                    self.parameters, element_params, coord, i, self.domain_size)
            elif self.parameters['element_type'] == "CRBeam":
                from source.element.cr_beam_element import CRBeamElement
                e = CRBeamElement(self.parameters, element_params,
                                  coord, i, self.domain_size)
            else:
                err_msg = "The requested element type \"" + \
                          self.parameters["element_type"]
                err_msg += "\" is not available \n"
                err_msg += "Choose one of: \"Bernoulli\", \"Timoshenko\", \"CRBeam\"\n"
                raise Exception(err_msg)
            self.elements.append(e)

        self.initialize_reference_coordinate()

    def update_equivalent_nodal_mass(self):
        '''
        The mass matrix for the beam element can be
        lumped or consistent

        For the equivalent nodal mass a lumped distribution
        is assumned

        Here only element and NO point mass contribution
        '''
        self.parameters['m'] = [0 for val in self.parameters['x']]
        # point mass contribution are initalized 
        self.parameters['point_m'] = [0 for val in self.parameters['x']]

        for idx in range(len(self.elements)):
            self.parameters['m'][idx] += 0.5 * self.elements[idx].A * self.elements[idx].rho * self.elements[idx].L
            self.parameters['m'][idx+1] += 0.5 * self.elements[idx].A * self.elements[idx].rho * self.elements[idx].L


    def initialize_reference_coordinate(self):
        self.nodal_coordinates["x0"] = list(
            e.ReferenceCoords[0] for e in self.elements)
        self.nodal_coordinates["x0"].append(
            self.elements[-1].ReferenceCoords[3])
        self.nodal_coordinates["x0"] = np.asarray(self.nodal_coordinates["x0"])
        self.nodal_coordinates["y0"] = list(
            e.ReferenceCoords[2] for e in self.elements)
        self.nodal_coordinates["y0"].append(
            self.elements[-1].ReferenceCoords[4])
        self.nodal_coordinates["y0"] = np.asarray(self.nodal_coordinates["y0"])
        self.nodal_coordinates["z0"] = list(
            e.ReferenceCoords[4] for e in self.elements)
        self.nodal_coordinates["z0"].append(
            self.elements[-1].ReferenceCoords[5])
        self.nodal_coordinates["z0"] = np.asarray(self.nodal_coordinates["z0"])

    def initialize_element_geometric_parameters(self, i):
        element_params = {}
        # element properties
        x = self.parameters['x_mid'][i]
        # area
        element_params['a'] = self.evaluate_characteristic_on_interval(
            x, 'c_a')

        # effective area of shear
        element_params['asy'] = self.evaluate_characteristic_on_interval(
            x, 'c_a_sy')
        element_params['asz'] = self.evaluate_characteristic_on_interval(
            x, 'c_a_sz')
        # second moment of inertia
        element_params['iy'] = self.evaluate_characteristic_on_interval(
            x, 'c_iy')
        element_params['iz'] = self.evaluate_characteristic_on_interval(
            x, 'c_iz')
        # torsion constant
        element_params['it'] = self.evaluate_characteristic_on_interval(
            x, 'c_it')
        return element_params

    def apply_elastic_bcs(self):
        # handle potential elastic BCs
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
                # using dict syntax -> for one dof entry onle one value is permitted
                self.point_stiffness[val] = elastic_bc_dofs_tmp[key]

    def apply_bcs(self):
        # TODO: make BC handling cleaner and compact
        bc = '\"' + \
             self.parameters["boundary_conditions"] + '\"'
        if bc in GD.AVAILABLE_BCS:
            # NOTE: create a copy of the list - useful if some parametric study is done
            self.bc_dofs = GD.BC_DOFS[self.domain_size][bc][:]
        else:
            err_msg = "The BC for input \"" + \
                      self.parameters["boundary_conditions"]
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join(GD.AVAILABLE_BCS)
            raise Exception(err_msg)

        self.apply_elastic_bcs()

        # list copy by slicing -> [:] -> to have a copy by value
        bc_dofs_global = self.bc_dofs[:]
        for idx, dof in enumerate(bc_dofs_global):
            # shift to global numbering the negative values
            if dof < 0:
                bc_dofs_global[idx] = dof + len(self.all_dofs_global)

        # only take bc's of interest
        self.dofs_to_keep = list(
            set(self.all_dofs_global) - set(bc_dofs_global))

    def update_outrigger_contribution(self):

        # point stiffness and point masses at respective dof for the outrigger
        for values in self.parameters['intervals']:
            if  values['m'] is not None:
                if values['bounds'][1] == "End":
                    geom_location = self.parameters['lx']
                    height_of_interval = geom_location
                else:
                    geom_location = values['bounds'][1]
                    height_of_interval = geom_location - values['bounds'][0]

                print('Outrigger at\n')
                print(' geometric location ', str(geom_location))

                geom_node_id = int(geom_location / self.lx_i)
                print(' geometric node id ', str(geom_node_id))

                # Point mass entries 
                # existing nodal mass from area, length and density
                existing_nodal_mass = self.parameters['m'][geom_node_id]

                # TODO more robust way, currently only makes sense with many elements
                # such that the outrigger as point mass is higher
                # so an increment can be added
                # For few element the target outrigger value will be too low

                if existing_nodal_mass < values['m']:
                    # increment needed to reach target nodal mass
                    incr_fctr = existing_nodal_mass / values['m'] 
                    self.parameters['point_m'][geom_node_id] += (1-incr_fctr)*values['m'] 
                else:
                    msg = "Existing nodal mass value of " + str(existing_nodal_mass) + " [kg]\n"
                    msg += "at location x= " + str(geom_location) + " [m] along the beam\n"
                    msg += "larger than target outrigger of " + str (values['m']) + " [kg].\n"
                    msg += "Not incrementing to target (as is it lower) but adding up to existing.\n"
                    print(msg)

                    self.parameters['point_m'][geom_node_id] += 0 # values['m']  DONE : check
                    self.parameters['m'][geom_node_id] += values['m']

                global_node_id = geom_node_id * GD.DOFS_PER_NODE[self.domain_size]
                affected_dof_ids = {}
                target_dof_vals = {}
                for idx, label in enumerate(GD.DOF_LABELS[self.domain_size]):
                    affected_dof_ids[label] = global_node_id + idx
                    target_dof_vals[label] = values['m']
                    if label in ['a', 'b', 'g']:
                        target_dof_vals[label] *= self.charact_length**2

                    # affects only diagonal entries for translation and rotation
                    # only global mass matrix entries
                    if existing_nodal_mass < values['m']:
                        # difference to target will be added
                        self.point_mass[global_node_id + idx] = (1-incr_fctr)*target_dof_vals[label]
                    else:
                        # target will be added as is
                        # AK : check
                        self.point_mass[global_node_id + idx] = 0
                        # self.point_mass[global_node_id + idx] = target_dof_vals[label]
                # Point stiffness entries
                
                outrigger_stiffness_ratio_y = values['out_stif_y']
                outrigger_stiffness_ratio_z = values['out_stif_z']

                point_stiffness_rotation_y = self.parameters['e'] * values['c_iy'][0] / height_of_interval * outrigger_stiffness_ratio_y
                point_stiffness_rotation_z = self.parameters['e'] * values['c_iz'][0] / height_of_interval * outrigger_stiffness_ratio_z
                point_stiffness_torsion = 0.0 # no torsional stiffness by addition of an outrigger system 

                global_node_id = geom_node_id * GD.DOFS_PER_NODE[self.domain_size]
                for idx, label in enumerate(GD.DOF_LABELS[self.domain_size]):
                    if label == 'a':
                        self.point_stiffness[global_node_id + idx] = point_stiffness_rotation_y  
                    elif label == 'b':
                        self.point_stiffness[global_node_id + idx] = point_stiffness_rotation_z  
                    elif label == 'g':
                        self.point_stiffness[global_node_id + idx] = point_stiffness_torsion  



    def calculate_total_mass(self, print_to_console=False):
        # update influencing parameters
        self.update_equivalent_nodal_mass()
        self.update_outrigger_contribution()
        # adding the point mass entries to this 
        print(self.parameters['m'])
        self.parameters['m_tot'] = 0.0
        for val in self.parameters['m']:
            self.parameters['m_tot'] += val
        # Done: Add outrigger masses to this entry as the parameters['m '] is 
        # already updated with the point masses in calculate_global matrices
        if print_to_console:
            print('CURRENT:')
            print('total mass ', self.parameters['m_tot'])
            print('density: ', self.parameters['rho'])
            print()

    def update_stiffness_matrix(self):
        k = self._get_stiffness()
        comp_k = self.apply_bc_by_reduction(k)
        return comp_k

    def calculate_global_matrices(self):
        # using computational values for m,b,k as this reduction is done otherwise many times
        # also update the outrigger contribution
        self.update_outrigger_contribution()
        # mass matrix
        self.m = self._get_mass()
        self.comp_m = self.apply_bc_by_reduction(self.m)
        # stiffness matrix
        self.k = self._get_stiffness()
        self.comp_k = self.apply_bc_by_reduction(self.k)
        # damping matrix - needs to be done after mass and stiffness as Rayleigh method nees these
        self.b = self._get_damping()
        self.comp_b = self.apply_bc_by_reduction(self.b)
        # updating the eleemnt mass contribution 
        for idx in range(len(self.parameters['x'])):
            self.parameters['m'][idx] += self.parameters['point_m'][idx] 



    def decompose_and_quantify_eigenmodes(self, considered_modes=15):
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.dofs_to_keep)
        else:
            if considered_modes > len(self.dofs_to_keep):
                considered_modes = len(self.dofs_to_keep)

        self.eigenvalue_solve()

        self.decomposed_eigenmodes = {'values': [], 'rel_contribution': [], 'eff_modal_mass': [],
                                      'rel_participation': []}

        for mode_idx in range(considered_modes):
            decomposed_eigenmode = {}
            rel_contrib = {}
            eff_modal_mass = {}
            rel_participation = {}
            selected_mode = self.eig_freqs_sorted_indices[mode_idx]

            for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.domain_size])),
                                  GD.DOF_LABELS[self.domain_size]):
                start = idx
                step = GD.DOFS_PER_NODE[self.domain_size]
                stop = self.eigen_modes_raw.shape[0] + idx - step
                decomposed_eigenmode[label] = self.eigen_modes_raw[start:stop +
                                                                         1:step][:, selected_mode]
                if label in ['a', 'b', 'g']:
                    # for rotation dofs multiply with a characteristic length
                    # to make comparable to translation dofs
                    rel_contrib[label] = self.charact_length * \
                                         linalg.norm(decomposed_eigenmode[label])
                else:
                    # for translation dofs
                    rel_contrib[label] = linalg.norm(
                        decomposed_eigenmode[label])

                # adding computation of modal mass
                # according to D-67: http://www.vibrationdata.com/tutorials2/beam.pdf
                # TODO: for now using element mass (as constant) and nodal dof value - make consistent
                # IMPORTANT
                if label in ['x', 'y', 'z', 'a']:
                    if rel_contrib[label] > GD.THRESHOLD:
                        eff_modal_numerator = 0.0
                        eff_modal_denominator = 0.0
                        total_mass = 0.0

                        for el_idx in range(self.n_elements-1):
                            # equivalent mass at node taken as average of 2 elements below and above node
                            storey_mass = (self.parameters['m'][el_idx] + self.parameters['m'][el_idx+1])/2
                            if label == 'a':
                                # NOTE for torsion using the equivalency of a rectangle with sides ly_i, lz_i
                                storey_mass *= (self.parameters['lz'][el_idx] ** 2 + self.parameters['ly'][el_idx] ** 2) / 12

                                # TODO check as torsion 4-5-6 does not seem to be ok in the results

                            total_mass += storey_mass

                            # taking the modal dof value at the node misusing naming el_idx
                            eff_modal_numerator += storey_mass * decomposed_eigenmode[label][el_idx]
                            eff_modal_denominator += storey_mass * decomposed_eigenmode[label][el_idx] ** 2

                        eff_modal_mass[label] = eff_modal_numerator ** 2 / eff_modal_denominator
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
            self.decomposed_eigenmodes['rel_participation'].append(
                rel_participation)

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

            for case_id in GD.MODE_CATEGORIZATION[self.domain_size]:
                match_for_case_id = False

                for dof_contribution_id in GD.MODE_CATEGORIZATION[self.domain_size][case_id]:
                    if self.decomposed_eigenmodes['rel_contribution'][i][dof_contribution_id] > GD.THRESHOLD:
                        match_for_case_id = True

                # TODO: check if robust enough for modes where 2 DoFs are involved
                if match_for_case_id:

                    if case_id in self.mode_identification_results:
                        # using list - so that results are ordered
                        self.mode_identification_results[case_id].append({
                            'mode_id' : (selected_mode + 1),
                            'eff_modal_mass': max(self.decomposed_eigenmodes['eff_modal_mass'][i].values()),
                            'rel_participation' : max(self.decomposed_eigenmodes['rel_participation'][i].values())
                        })
                    else:
                        # using list - so that results are ordered
                        self.mode_identification_results[case_id] = [{
                            'mode_id' : (selected_mode + 1),
                            'eff_modal_mass': max(self.decomposed_eigenmodes['eff_modal_mass'][i].values()),
                            'rel_participation' : max(self.decomposed_eigenmodes['rel_participation'][i].values())
                        }]

        if print_to_console:
            print('Result of decoupled eigenmode identification for the first ' +
                  str(considered_modes) + ' mode(s)')

            for mode_type, type_results in self.mode_identification_results.items():
                print('  Mode type:', mode_type)
                for t_res in type_results:
                    m_id = t_res['mode_id']
                    # will have: mode_id, eff_modal_mass, rel_participation
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
                if val['bounds'][0] <= running_coord < val['bounds'][1]:
                    return evaluate_polynomial(running_coord - val['bounds'][0], val[characteristic_identifier])
            elif "End" in val['bounds']:
                if val['bounds'][0] <= running_coord <= self.parameters['lx']:
                    return evaluate_polynomial(running_coord - val['bounds'][0], val[characteristic_identifier])

    def write_properties(self, global_folder_path):
        lines = []

        for idx, elem in enumerate(self.elements):
            lines.append([str(idx),
                          '{:.3f}'.format(self.nodal_coordinates["x0"][idx]),
                          '{:.3f}'.format(self.nodal_coordinates["x0"][idx + 1]),
                          '{:.3f}'.format(self.parameters['x_mid'][idx]),
                          '{:.3f}'.format(self.parameters['ly'][idx]),
                          '{:.3f}'.format(self.parameters['lz'][idx]),
                          '{:.3f}'.format(elem.A),
                          '{:.3f}'.format(elem.Asy),
                          '{:.3f}'.format(elem.Asz),
                          '{:.3f}'.format(elem.Iy),
                          '{:.3f}'.format(elem.Iz),
                          '{:.3f}'.format(elem.It),
                          '{:.3f}'.format(elem.Ip),
                          '{:.3f}'.format(elem.Py),
                          '{:.3f}'.format(elem.Pz),
                          # NOTE element and NOT nodal mass
                          '{:.3f}'.format(elem.A * elem.rho * elem.L)
                          ])

        file_header = '# Properties of the structure model\n'
        file_header += '# ElemNr |  x_start [m] | x_end [m] | x_mid [m] | '
        file_header += 'Cross section ly [m] | Cross section lz [m] | '
        file_header += 'Area [m^2] | Shear area_sy  [m^2] | Shear area_sz  [m^2] | '
        file_header += 'Moment of inertia Iy [m^4] | Moment of inertia Iz [m^4] | '
        file_header += 'Torsion constant It  [m^4] | Polar moment of inertia Ip [m^4] | '
        file_header += 'Relative shear factor Py  [-] | Relative shear factor Pz  [-] | '
        file_header += 'Mass m  [kg] - Total mass ' + '{:.3f}'.format(self.parameters['m_tot']) + ' [kg]\n'

        file_name = 'structure_model_properties.dat'

        writer_utilities.write_table(os_join(global_folder_path, file_name),
                                     file_header,
                                     lines)

    def plot_properties(self, pdf_report, display_plot):

        plot_title = []
        struct_property_data = []
        plot_legend = []
        plot_style = []

        #
        plot_title.append("Cross section length over running coordinate x")
        struct_property_data.append([{'x': self.parameters['x'], 'y': self.parameters['ly']},
                                     {'x': self.parameters['x'], 'y': self.parameters['lz']}])
        plot_legend.append(['ly [m]', 'lz[m]'])
        plot_style.append(['-ko', '--ro'])

        #
        plot_title.append("Area(s) over running coordinate x")
        struct_property_data.append([{'x': self.parameters['x_mid'], 'y': [elem.A for elem in self.elements]},
                                     {'x': self.parameters['x_mid'], 'y': [
                                         elem.Asy for elem in self.elements]},
                                     {'x': self.parameters['x_mid'], 'y': [elem.Asz for elem in self.elements]}])
        plot_legend.append(['A [m^2]', 'Asy [m^2]', 'Asz [m^2]'])
        plot_style.append(['-ko', '--ro', '-.bo'])

        #
        plot_title.append("Moment(s) of inertia over running coordinate x")
        struct_property_data.append([{'x': self.parameters['x_mid'], 'y': [elem.Iy for elem in self.elements]},
                                     {'x': self.parameters['x_mid'], 'y': [elem.Iz for elem in self.elements]}])
        plot_legend.append(['Iy [m^4]', 'Iz [m^4]'])
        plot_style.append(['-ko', '--ro'])

        #
        plot_title.append("Torsional properties over running coordinate x")
        struct_property_data.append([{'x': self.parameters['x_mid'], 'y': [elem.It for elem in self.elements]},
                                     {'x': self.parameters['x_mid'], 'y': [elem.Ip for elem in self.elements]}])
        plot_legend.append(['It [m^4]', 'Ip [m^4]'])
        plot_style.append(['-ko', '--ro'])

        #
        plot_title.append(
            "Relative importance of shear over running coordinate x")
        struct_property_data.append([{'x': self.parameters['x_mid'], 'y': [elem.Py for elem in self.elements]},
                                     {'x': self.parameters['x_mid'], 'y': [elem.Pz for elem in self.elements]}])
        plot_legend.append(['Py [-]', 'Pz [-]'])
        plot_style.append(['-ko', '--ro'])

        #
        plot_title.append("Nodal mass over running coordinate x - Total mass " + '{:.3f}'.format(self.parameters['m_tot']) + " [kg]")
        struct_property_data.append([{'x': self.parameters['x'], 'y': self.parameters['m']}])
        plot_legend.append(['m [kg]'])
        plot_style.append(['-ko'])

        for idx in range(len(plot_title)):
            plotter_utilities.plot_properties(pdf_report,
                                              display_plot,
                                              plot_title[idx],
                                              struct_property_data[idx],
                                              plot_legend[idx],
                                              plot_style[idx])

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
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, np.arange(matrix.shape[1]))
        elif axis == 'column':
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.dofs_to_keep)
        elif axis == 'both':
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)
        elif axis == 'row_vector':
            ixgrid = np.ix_(self.dofs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])
        elif axis == 'column_vector':
            ixgrid = np.ix_(self.dofs_to_keep)
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
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(self.all_dofs_global)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), self.dofs_to_keep)
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'both':
            rows = len(self.all_dofs_global)
            cols = rows
            # make a grid of indices on interest
            ixgrid = np.ix_(self.dofs_to_keep, self.dofs_to_keep)
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'row_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.dofs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'column_vector':
            rows = len(self.all_dofs_global)
            cols = 1
            ixgrid = np.ix_(self.dofs_to_keep)
            extended_matrix = np.zeros((rows,))
        else:
            err_msg = "The extension mode with input \"" + axis
            err_msg += "\" for axis is not avaialbe \n"
            err_msg += "Choose one of: \"row\", \"column\", \"both\", \"row_vector\""
            raise Exception(err_msg)

        extended_matrix[ixgrid] = matrix

        return extended_matrix

    # NOTE: not used for now
    def _assemble_el_into_glob(self, el_matrix):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matrix entries
        for i in range(self.parameters['n_el']):
            glob_matrix[
            GD.DOFS_PER_NODE[self.domain_size] * i: GD.DOFS_PER_NODE[self.domain_size] * i +
                                                    GD.DOFS_PER_NODE[
                                                        self.domain_size] * GD.NODES_PER_LEVEL,
            GD.DOFS_PER_NODE[self.domain_size] * i: GD.DOFS_PER_NODE[self.domain_size] * i +
                                                    GD.DOFS_PER_NODE[
                                                        self.domain_size] * GD.NODES_PER_LEVEL] += el_matrix
        return glob_matrix

    def _get_mass(self):

        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))

        # fill global mass matrix entries
        for element in self.elements:
            el_matrix = element.get_element_mass_matrix()
            i_start = GD.DOFS_PER_NODE[self.domain_size] * element.index
            i_end = i_start + GD.DOFS_PER_NODE[self.domain_size] * GD.NODES_PER_LEVEL
            glob_matrix[
                i_start: i_end,
                i_start: i_end
            ] += el_matrix

        for idx, val in self.point_mass.items():
            glob_matrix[idx, idx] += val

        return glob_matrix

    def _get_stiffness(self):
        # global stiffness matrix initialization with zeros
        glob_matrix = np.zeros((self.n_nodes * GD.DOFS_PER_NODE[self.domain_size],
                                self.n_nodes * GD.DOFS_PER_NODE[self.domain_size]))

        # fill global stiffness matrix entries
        for element in self.elements:
            el_matrix = element.get_element_stiffness_matrix()
            i_start = GD.DOFS_PER_NODE[self.domain_size] * element.index
            i_end = i_start + GD.DOFS_PER_NODE[self.domain_size] * GD.NODES_PER_LEVEL

            glob_matrix[
            i_start: i_end,
            i_start: i_end] += el_matrix

        for idx, val in self.point_stiffness.items():
            glob_matrix[idx, idx] += val

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

        self.rayleigh_coefficients = np.linalg.solve(0.5 *
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
        return self.rayleigh_coefficients[0] * self.m + self.rayleigh_coefficients[1] * self.k
