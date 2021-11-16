import numpy as np
import json
from os.path import join as os_join

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.writer_utilitites as writer_utilities
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
from source.auxiliary.other_utilities import get_adjusted_path_string
import source.auxiliary.global_definitions as GD


class StaticAnalysis(AnalysisType):
    """
    Dervied class for the static analysis of a given structure model
    """

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "type": "static_analysis",
        "settings": {},
        "input": {},
        "output": {}}

    def __init__(self, structure_model, parameters):

        # validating and assign model parameters
        validate_and_assign_defaults(
            StaticAnalysis.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        super().__init__(structure_model, self.parameters["type"])

        # TODO include some specifiers in the parameters, do not hard code
        if get_adjusted_path_string(self.parameters['input']['file_path']) == get_adjusted_path_string('some/path'):
            err_msg = get_adjusted_path_string(
                self.parameters['input']['file_path'])
            err_msg += " is not a valid file!"
            raise Exception(err_msg)
        else:
            print(get_adjusted_path_string(
                self.parameters['input']['file_path']) + ' set as load file path in StaticAnalysis')
            if self.parameters['input']['is_time_history_file']:
                self.force = np.load(get_adjusted_path_string(self.parameters['input']['file_path']))[
                    :, self.parameters['input']['selected_time_step']]
                self.force = np.reshape(self.force, (-1,1))
            else:
                self.force = np.load(get_adjusted_path_string(
                    self.parameters['input']['file_path']))

        # of nodes-dofs
        n_dofs_model = structure_model.n_nodes * \
            GD.DOFS_PER_NODE[structure_model.domain_size]
        n_dofs_force = len(self.force)
        if n_dofs_model != n_dofs_force:
            err_msg = "The number of the degrees of freedom " + \
                str(n_dofs_model) + " of the structural model\n"
            err_msg += "does not match the degrees of freedom " + \
                str(n_dofs_force) + " of the load time history\n"
            err_msg += "specified in \"runs\" -> for \"type\":\"static_analysis\" -> \"input\" -> \"file_path\"!\n"
            err_msg += "The structural model has:\n"
            err_msg += "   " + \
                str(structure_model.n_elements) + " number of elements\n"
            err_msg += "   " + \
                str(structure_model.n_nodes) + " number of nodes\n"
            err_msg += "   " + str(n_dofs_model) + " number of dofs.\n"
            err_msg += "The naming of the force time history should reflect the number of nodes\n"
            err_msg += "using the convention \"<force_type>_force_<n_nodes>_nodes.npy\"\n"
            digits_in_filename = [
                s for s in self.parameters['input']['file_path'].split('_') if s.isdigit()]
            if len(digits_in_filename) == 1:
                err_msg += "where currently <n_nodes> = " + \
                    digits_in_filename[0] + " (separated by underscores)!"
            else:
                err_msg += "but found multiple digits: " + \
                    ', '.join(digits_in_filename) + \
                    " (separated by underscores)!"
            raise Exception(err_msg)

    def solve(self):
        print("Solving for ext_force in StaticAnalysis derived class \n")
        # self.force = ext_force
        force = self.structure_model.apply_bc_by_reduction(
            self.force, 'row_vector')

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)

        static_result = np.linalg.solve(k, force)
        self.static_result = self.structure_model.recuperate_bc_by_extension(
            static_result, 'row_vector')

        f = np.dot(self.structure_model.k, self.static_result)
        self.resisting_force = self.force - f

        # nullify
        self.resisting_force[abs(self.resisting_force) < GD.THRESHOLD] = 0.0

        # placeholders
        self.force_action = {"x": np.zeros(0),
                        "y": np.zeros(0),
                        "z": np.zeros(0),
                        "a": np.zeros(0),
                        "b": np.zeros(0),
                        "g": np.zeros(0)}

        self.reaction = {"x": np.zeros(0),
                         "y": np.zeros(0),
                         "z": np.zeros(0),
                         "a": np.zeros(0),
                         "b": np.zeros(0),
                         "g": np.zeros(0)}

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.static_result.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.static_result[start:stop +
                                                                               1:step][:, 0]
            self.force_action[label] = self.force[start:stop + 1:step]
            self.reaction[label] = self.resisting_force[start:stop + 1:step][:, 0]

    def plot_solve_result(self, pdf_report, display_plot):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting result in StaticAnalysis \n")

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.static_result.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.static_result[start:stop +
                                                                               1:step][:, 0]
            self.force_action[label] = self.force[start:stop + 1:step]
            self.reaction[label] = self.resisting_force[start:stop + 1:step][:, 0]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"],
                                    self.structure_model.nodal_coordinates["a"]],
                    "deformed": None}

        force = {"external": [self.force_action["x"],
                              self.force_action["y"],
                              self.force_action["z"]],
                 "reaction": [self.reaction["x"],
                              self.reaction["y"],
                              self.reaction["z"]]}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Static Analysis : "

        plotter_utilities.plot_result(pdf_report,
                                      display_plot,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      1)

    def write_solve_result(self, global_folder_path):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting result in StaticAnalysis \n")

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.static_result.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.static_result[start:stop +
                                                                               1:step][:, 0]
            self.force_action[label] = self.force[start:stop + 1:step]
            self.reaction[label] = self.resisting_force[start:stop + 1:step][:, 0]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": [self.force_action["x"],
                              self.force_action["y"],
                              self.force_action["z"]],
                 "reaction": [self.reaction["x"],
                              self.reaction["y"],
                              self.reaction["z"]]}

        scaling = {"deformation": 1,
                   "force": 1}

        file_header = "# Static Analysis"

        file_name = 'static_analysis' + '.dat'

        writer_utilities.write_result(os_join(global_folder_path, file_name), file_header,
                                      geometry, scaling)

    def write_result_at_dof(self, global_folder_path, dof, selected_result):
        """
        Pass to plot function:
            Plots the time series of required quantitiy
        """
        print('Writing result for selected dof in StaticAnalysis \n')

        if selected_result == 'displacement':
            result_data = self.static_result[dof, :]
        elif selected_result == 'force':
            result_data = self.force[dof, :]
        elif selected_result == 'reaction':
            if dof in self.structure_model.bc_dofs or dof in self.structure_model.elastic_bc_dofs:
                result_data = self.resisting_force[dof, :]
            else:
                err_msg = "The selected DoF \"" + str(dof)
                err_msg += "\" is not avaialbe in the list of available boundary condition dofs \n"
                err_msg += "Choose one of: " + \
                           ", ".join([str(val)
                                      for val in self.structure_model.bc_dofs])
                raise Exception(err_msg)
        else:
            err_msg = "The selected result \"" + selected_result
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"displacement\", \"force_ext\", \"force_react\", \"reaction\""
            raise Exception(err_msg)

        coord_label = GD.DOF_LABELS[self.structure_model.domain_size][int(dof % GD.DOFS_PER_NODE[self.structure_model.domain_size])]

        file_header = "# Static Analysis result " + selected_result + "\n"
        file_header += "# for DoF " + str(dof) + " -> " + coord_label + " over time \n"
        # TODO add DoF height coordinate into header

        file_name = 'static_analysis_result_' + \
                    selected_result + '_for_dof_' + str(dof) + '.dat'

        writer_utilities.write_result_at_dof(os_join(global_folder_path, file_name),
                                             file_header,
                                             result_data,
                                             [0.0])

    def postprocess(self, global_folder_path, pdf_report, display_plot, skin_model_params):
        """
        Postprocess something
        """
        print("Postprocessing in StaticAnalysis derived class \n")

        for plot_result in self.parameters['output']['plot']:
            if plot_result == 'deformation':
                self.plot_solve_result(pdf_report, display_plot)
            if plot_result == 'forces':
                pass

        for write_result in self.parameters['output']['write']:
            if write_result == 'deformation':
                self.write_solve_result(global_folder_path)
            if write_result == 'forces':
                pass

        for idx_dof, dof_id in enumerate(self.parameters['output']['selected_dof']['dof_list']):
            for idx_res, res in enumerate(self.parameters['output']['selected_dof']['result_type'][idx_dof]):
                if res in ['displacement', 'force','reaction']:
                    # if self.parameters['output']['selected_dof']['plot_result'][idx_dof][idx_res]:
                    #     self.plot_result_at_dof(
                    #         pdf_report, display_plots, dof_id, res)
                    if self.parameters['output']['selected_dof']['write_result'][idx_dof][idx_res]:
                        self.write_result_at_dof(
                            global_folder_path, dof_id, res)
                else:
                    err_msg = "The selected result \"" + res
                    err_msg += "\" is not avaialbe \n"
                    err_msg += "Choose one of: \"displacement\", \"force\" , \"reaction\""
                    raise Exception(err_msg)

        pass
