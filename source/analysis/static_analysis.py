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
        
        # if a static analysis is included and the number of nodes does not match the force file
        # TODO include some specifiers in the parameters, do not hard code
        if get_adjusted_path_string(self.parameters['input']['file_path']) == get_adjusted_path_string('some/path'):
            err_msg = get_adjusted_path_string(
                self.parameters['input']['file_path'])
            err_msg += " is not a valid file!"
            raise Exception(err_msg)
        else:
            #print(get_adjusted_path_string(
            #   self.parameters['input']['file_path']) + ' set as load file path in StaticAnalysis')
            if self.parameters['input']['is_time_history_file']:
                self.force = np.load(get_adjusted_path_string(self.parameters['input']['file_path']))[
                    :, self.parameters['input']['selected_time_step']]
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
        #print("Solving for ext_force in StaticAnalysis derived class \n")
        # self.force = ext_force
        force = self.structure_model.apply_bc_by_reduction(
            self.force, 'row_vector')

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)
        #print(k)
        self.static_result = np.linalg.solve(k, force)
        self.static_result = self.structure_model.recuperate_bc_by_extension(
            self.static_result, 'row_vector')
        self.force_action = {"x": np.zeros(0),
                             "y": np.zeros(0),
                             "z": np.zeros(0),
                             "a": np.zeros(0),
                             "b": np.zeros(0),
                             "g": np.zeros(0)}

        #self.force = self.structure_model.recuperate_bc_by_extension(self.force,'row_vector')
        self.resisting_force = self.force - np.dot(self.structure_model.k, self.static_result)
        ixgrid = np.ix_(self.structure_model.dofs_to_keep, [0])
        self.resisting_force[ixgrid] = 0
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


    def plot_solve_result_2D(self, pdf_report, display_plot):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting 2D result in StaticAnalysis \n")

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

        plotter_utilities.plot_result_2D(pdf_report,
                                         display_plot,
                                         plot_title,
                                         geometry,
                                         force,
                                         scaling)

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

    def postprocess(self, global_folder_path, pdf_report, display_plot, skin_model_params):
        """
        Postprocess something
        """
        print("Postprocessing in StaticAnalysis derived class \n")
        #print("in postprocess static analysis derived class switched display plot to TRUE \n")

        for plot_result in self.parameters['output']['plot']:
            if plot_result == 'deformation':
                #display_plot = True
                #self.plot_solve_result(pdf_report, display_plot)
                #display_plot = True
                self.plot_solve_result_2D(pdf_report, display_plot)
            if plot_result == 'forces':
                pass

        for write_result in self.parameters['output']['write']:
            if write_result == 'deformation':
                self.write_solve_result(global_folder_path)
            if write_result == 'forces':
                pass

        pass
