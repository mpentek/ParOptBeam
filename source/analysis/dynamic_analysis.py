from os.path import join as os_join

import numpy as np

import source.auxiliary.global_definitions as GD
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.visualize_skin_model_utilities as visualize_skin_model_utilities
import source.postprocess.writer_utilitites as writer_utilities
from source.analysis.analysis_type import AnalysisType
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults


def transform_into_modal_coordinates(modal_transform_matrix, matrix):
    return np.matmul(np.matmul(np.transpose(modal_transform_matrix), matrix), modal_transform_matrix)


class DynamicAnalysis(AnalysisType):
    """
    Derived class for the dynamic analysis of a given structure model

    """

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "type": "dynamic_analysis",
        "settings": {},
        "input": {},
        "output": {}}

    def __init__(self, structure_model, parameters):

        # validating and assign model parameters
        validate_and_assign_defaults(
            DynamicAnalysis.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        # time parameters
        time_integration_scheme = self.parameters['settings']['time']['integration_scheme']
        start = self.parameters['settings']['time']['start']
        stop = self.parameters['settings']['time']['end']
        # TODO check if this is the correct way
        self.dt = self.parameters['settings']['time']['step']
        steps = int((self.parameters['settings']['time']['end']
                     - self.parameters['settings']['time']['start']) / self.dt) + 1
        self.array_time = np.linspace(start, stop, steps)

        # TODO include some specifiers in the parameters, do not hard code
        if get_adjusted_path_string(self.parameters['input']['file_path']) == get_adjusted_path_string('some/path'):
            err_msg = get_adjusted_path_string(
                self.parameters['input']['file_path'])
            err_msg += " is not a valid file!"
            raise Exception(err_msg)
        else:
            print(get_adjusted_path_string(
                self.parameters['input']['file_path']) + ' set as load file path in DynamicAnalysis')
            force = np.load(get_adjusted_path_string(
                self.parameters['input']['file_path']))

        super().__init__(structure_model, self.parameters["type"])
        # print("Force: ", len(force))
        # overwriting attribute from base constructors
        self.force = force

        # check dimensionality
        # of time
        len_time_gen_array = len(self.array_time)
        len_time_force = len(self.force[0])
        if len_time_gen_array != len_time_force:
            err_msg = "The length " + \
                str(len_time_gen_array) + \
                " of the time array generated based upon parameters\n"
            err_msg += "specified in \"runs\" -> for \"type\":\"dynamic_analysis\" -> \"settings\" -> \"time\"\n"
            err_msg += "does not match the time series length " + \
                str(len_time_force) + " of the load time history\n"
            err_msg += "specified in \"runs\" -> for \"type\":\"dynamic_analysis\" -> \"input\" -> \"file_path\"!\n"
            raise Exception(err_msg)

        # of nodes-dofs
        n_dofs_model = structure_model.n_nodes * \
            GD.DOFS_PER_NODE[structure_model.domain_size]
        n_dofs_force = len(self.force)
        if n_dofs_model != n_dofs_force:
            err_msg = "The number of the degrees of freedom " + \
                str(n_dofs_model) + " of the structural model\n"
            err_msg += "does not match the degrees of freedom " + \
                str(n_dofs_force) + " of the load time history\n"
            err_msg += "specified in \"runs\" -> for \"type\":\"dynamic_analysis\" -> \"input\" -> \"file_path\"!\n"
            err_msg += "The structural model has:\n"
            err_msg += "   " + \
                str(structure_model.n_elements) + " number of elements\n"
            err_msg += "   " + \
                str(structure_model.n_nodes) + " number of nodes\n"
            err_msg += "   " + str(n_dofs_model) + " number of dofs.\n"
            err_msg += "The naming of the force time history should reflect the number of nodes\n"
            err_msg += "using the convention \"dynamic_force_<n_nodes>_nodes.npy\"\n"
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

        rows = len(self.structure_model.apply_bc_by_reduction(
            self.structure_model.k))

        # initial condition of zero displacement and velocity used for the time being.
        # TODO : to incorporate user defined initial conditions

        u0 = np.zeros(rows)  # initial displacement
        v0 = np.zeros(rows)  # initial velocity
        a0 = np.zeros(rows)  # initial acceleration
        initial_conditions = np.array([u0, v0, a0])

        if 'run_in_modal_coordinates' in self.parameters['settings']:
            if self.parameters['settings']['run_in_modal_coordinates']:
                self.transform_into_modal = True
            else:
                self.transform_into_modal = False
                pass
        else:
            self.transform_into_modal = False
            pass

        # TODO check if concept of comp - computational model is robust and generic enough
        self.comp_m = np.copy(self.structure_model.comp_m)
        self.comp_k = np.copy(self.structure_model.comp_k)
        self.comp_b = np.copy(self.structure_model.comp_b)

        # tranformation to the modal coordinates
        if self.transform_into_modal:
            self.comp_m = transform_into_modal_coordinates(
                self.structure_model.eigen_modes_raw, self.comp_m)
            self.comp_b = transform_into_modal_coordinates(
                self.structure_model.eigen_modes_raw, self.comp_b)
            self.comp_k = transform_into_modal_coordinates(
                self.structure_model.eigen_modes_raw, self.comp_k)

        if force.shape[1] != len(self.array_time):
            err_msg = "The time step for forces does not match the time step defined"
            raise Exception(err_msg)

        # external forces
        force = self.structure_model.apply_bc_by_reduction(self.force, 'row')

        if self.transform_into_modal:
            force = np.dot(np.transpose(
                self.structure_model.eigen_modes_raw), force)

        print(self.parameters)
        if self.parameters["settings"]["solver_type"] == "Linear":
            from source.solving_strategies.strategies.linear_solver import LinearSolver
            self.solver = LinearSolver(self.array_time, time_integration_scheme, self.dt,
                                       [self.comp_m, self.comp_b, self.comp_k],
                                       initial_conditions, force,
                                       self.structure_model)
        elif self.parameters["settings"]["solver_type"] == "Picard":
            from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
            self.solver = ResidualBasedPicardSolver(self.array_time, time_integration_scheme, self.dt,
                                                    [self.comp_m, self.comp_b,
                                                        self.comp_k],
                                                    initial_conditions, force,
                                                    self.structure_model)
        elif self.parameters["settings"]["solver_type"] == "NewtonRaphson":
            from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
            self.solver = ResidualBasedNewtonRaphsonSolver(self.array_time, time_integration_scheme, self.dt,
                                                           [self.comp_m, self.comp_b,
                                                               self.comp_k],
                                                           initial_conditions, force,
                                                           self.structure_model)
        else:
            err_msg = "The requested solver type \"" + \
                self.parameters["settings"]["solver_type"]
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"Linear\", \"Picard\", \"NewtonRaphson\"\n"
            raise Exception(err_msg)

    def solve(self):

        print("Solving the structure for dynamic loads \n")
        self.solver.solve()

        # transforming back to normal coordinates :
        if self.transform_into_modal:
            self.solver.displacement = np.matmul(
                self.structure_model.eigen_modes_raw, self.solver.displacement)
            self.solver.velocity = np.matmul(
                self.structure_model.eigen_modes_raw, self.solver.velocity)
            self.solver.acceleration = np.matmul(
                self.structure_model.eigen_modes_raw, self.solver.acceleration)

        self.solver.displacement = self.structure_model.recuperate_bc_by_extension(
            self.solver.displacement)
        self.solver.velocity = self.structure_model.recuperate_bc_by_extension(
            self.solver.velocity)
        self.solver.acceleration = self.structure_model.recuperate_bc_by_extension(
            self.solver.acceleration)
                

        f1 = np.matmul(self.structure_model.m, self.solver.acceleration)
        f2 = np.matmul(self.structure_model.b, self.solver.velocity)
        f3 = np.matmul(self.structure_model.k, self.solver.displacement)
        self.solver.dynamic_reaction = self.force - f1 - f2 - f3

        # TODO : elastic support reaction computation 

    def plot_result_at_dof(self, pdf_report, display_plots, dof, selected_result):
        """
        Pass to plot function:
            Plots the time series of required quantitiy 
        """
        print('Plotting result for selected dof in dynamic analysis \n')

        coord_label = GD.DOF_LABELS[self.structure_model.domain_size][int(dof % GD.DOFS_PER_NODE[self.structure_model.domain_size])]

        plot_title = selected_result.capitalize() + ' at DoF ' + str(dof) + " -> " + coord_label
        if selected_result == 'displacement':
            result_data = self.solver.displacement[dof, :]
        elif selected_result == 'velocity':
            result_data = self.solver.velocity[dof, :]
        elif selected_result == 'acceleration':
            result_data = self.solver.acceleration[dof, :]
        elif selected_result == 'reaction':
            if dof in self.structure_model.bc_dofs or dof in self.structure_model.elastic_bc_dofs:
                result_data = self.solver.dynamic_reaction[dof, :]
            else:
                err_msg = "The selected DoF \"" + str(dof)
                err_msg += "\" is not avaialbe in the list of available boundary condition dofs \n"
                err_msg += "Choose one of: " + \
                           ", ".join([str(val)
                                      for val in self.structure_model.bc_dofs])
                raise Exception(err_msg)
        else:
            err_msg = "The selected result \"" + selected_result
            err_msg += "\" is not avaialbe \n"
            err_msg += "Choose one of: \"displacement\", \"velocity\", \"acceleration\", \"reaction\""
            raise Exception(err_msg)

        plotter_utilities.plot_dynamic_result(pdf_report,
                                              display_plots,
                                              plot_title,
                                              result_data,
                                              self.array_time)

    def write_result_at_dof(self, global_folder_path, dof, selected_result):
        """
        Pass to plot function:
            Plots the time series of required quantitiy 
        """
        print('Writing result for selected dof in DynamicAnalysis \n')

        if selected_result == 'displacement':
            result_data = self.solver.displacement[dof, :]
        elif selected_result == 'velocity':
            result_data = self.solver.velocity[dof, :]
        elif selected_result == 'acceleration':
            result_data = self.solver.acceleration[dof, :]
        elif selected_result == 'reaction':
            if dof in self.structure_model.bc_dofs or dof in self.structure_model.elastic_bc_dofs:
                result_data = self.solver.dynamic_reaction[dof, :]
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
            err_msg += "Choose one of: \"displacement\", \"velocity\", \"acceleration\", \"reaction\""
            raise Exception(err_msg)

        coord_label = GD.DOF_LABELS[self.structure_model.domain_size][int(dof % GD.DOFS_PER_NODE[self.structure_model.domain_size])]

        file_header = "# Dynamic Analysis result " + selected_result + "\n"
        file_header += "# for DoF " + str(dof) + " -> " + coord_label + " over time \n"
        # TODO add DoF height coordinate into header

        file_name = 'dynamic_analysis_result_' + \
                    selected_result + '_for_dof_' + str(dof) + '.dat'

        writer_utilities.write_result_at_dof(os_join(global_folder_path, file_name),
                                             file_header,
                                             result_data,
                                             self.array_time)

    def plot_selected_time(self, pdf_report, display_plots, selected_time):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.solver.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")

        # find closet time step
        idx_time = np.where(self.array_time >= selected_time)[0][0]

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.solver.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                     1:step][:, idx_time]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,  # [np.append(origin_point, self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": None}  # [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Dyanimc Analyis: Deformation at t = " + \
                     str(selected_time) + " [s]"

        plotter_utilities.plot_result(pdf_report,
                                      display_plots,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      1)

    def write_selected_time(self, global_folder_path, selected_time):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.solver.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")

        # find closet time step
        idx_time = np.where(self.array_time >= selected_time)[0][0]

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.solver.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                     1:step][:, idx_time]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,  # [np.append(origin_point, self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": None}  # [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}

        file_header = "# Dyanimc Analyis: Deformation at t = " + \
                      str(selected_time) + " [s]" + "\n"

        file_name = 'dynamic_analysis_selected_time_' + \
                    str(selected_time) + 's.dat'

        writer_utilities.write_result(os_join(global_folder_path, file_name), file_header,
                                      geometry, scaling)

    def plot_selected_step(self, pdf_report, display_plots, selected_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.solver.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected step in DynamicAnalysis \n")

        # TODO refactor so that plot_selected_time calls plot_selected_step
        idx_time = selected_step

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.solver.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                     1:step][:, idx_time]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,  # [np.append(origin_point, self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": None}  # [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Dynamic Analysis: Deformation for step = " + str(idx_time) + " at t = " + \
                     str(self.array_time[idx_time]) + " [s]"

        plotter_utilities.plot_result(pdf_report,
                                      display_plots,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      1)

    def write_selected_step(self, global_folder_path, selected_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.solver.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected step in DynamicAnalysis \n")

        # TODO refactor so that plot_selected_time calls plot_selected_step
        idx_time = selected_step

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.solver.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                     1:step][:, idx_time]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,  # [np.append(origin_point, self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": None}  # [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}

        file_header = "# Dynamic Analysis: Deformation for step = " + str(idx_time) + " at t = " + \
                      str(self.array_time[idx_time]) + " [s]"

        file_name = 'dynamic_analysis_selected_step_' + str(idx_time) + '.dat'

        writer_utilities.write_result(os_join(global_folder_path, file_name), file_header,
                                      geometry, scaling)

    def animate_time_history(self):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.solver.displacement -> here as time series  
        """

        print("Animating time history in DynamicAnalysis \n")
        print("Copying time step solution from solver")
        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.solver.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                     1:step]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Dyanimc Analyis: Deformation over time"

        plotter_utilities.animate_result(plot_title,
                                         self.array_time,
                                         geometry,
                                         force,
                                         scaling)

    def animate_skin_model_time_history(self, skin_model_params):
        print("Animating skin model time history")
        if not self.parameters['output']['animate_time_history']:
            print("Copying time step solution from solver")
            for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                                  GD.DOF_LABELS[self.structure_model.domain_size]):
                start = idx
                step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
                stop = self.solver.displacement.shape[0] + idx - step
                self.structure_model.nodal_coordinates[label] = self.solver.displacement[start:stop +
                                                                                         1:step]
        skin_model_params["result_path"] = os_join(
            "output", self.structure_model.name)
        skin_model_params["dynamic_analysis"] = {}
        skin_model_params["dynamic_analysis"]["time_step"] = self.parameters['settings']['time']['step']
        skin_model_params["dynamic_analysis"]["start_record"] = \
            self.parameters['output']['skin_model_animation_parameters']['start_record']
        skin_model_params["dynamic_analysis"]["end_record"] = \
            self.parameters['output']['skin_model_animation_parameters']['end_record']
        skin_model_params["dynamic_analysis"]["record_step"] = \
            self.parameters['output']['skin_model_animation_parameters']['record_step']
        skin_model_params["dofs_input"] = self.structure_model.nodal_coordinates

        visualize_skin_model_utilities.visualize_skin_model(skin_model_params)

    def postprocess(self, global_folder_path, pdf_report, display_plots, skin_model_params):
        """
        Postprocess something
        """
        print("Postprocessing in DynamicAnalysis derived class \n")

        for time in self.parameters['output']['selected_instance']['plot_time']:
            self.plot_selected_time(pdf_report, display_plots, time)

        for time in self.parameters['output']['selected_instance']['write_time']:
            self.write_selected_time(global_folder_path, time)

        for step in self.parameters['output']['selected_instance']['plot_step']:
            self.plot_selected_step(pdf_report, display_plots, step)

        for step in self.parameters['output']['selected_instance']['write_step']:
            self.write_selected_step(global_folder_path, step)

        if self.parameters['output']['animate_time_history']:
            self.animate_time_history()

        if skin_model_params is not None:
            if self.parameters['output']['animate_skin_model_time_history']:
                self.animate_skin_model_time_history(skin_model_params)

        for idx_dof, dof_id in enumerate(self.parameters['output']['selected_dof']['dof_list']):
            for idx_res, res in enumerate(self.parameters['output']['selected_dof']['result_type'][idx_dof]):
                if res in ['displacement', 'velocity', 'acceleration', 'reaction']:
                    if self.parameters['output']['selected_dof']['plot_result'][idx_dof][idx_res]:
                        self.plot_result_at_dof(
                            pdf_report, display_plots, dof_id, res)
                    if self.parameters['output']['selected_dof']['write_result'][idx_dof][idx_res]:
                        self.write_result_at_dof(
                            global_folder_path, dof_id, res)
                else:
                    err_msg = "The selected result \"" + res
                    err_msg += "\" is not avaialbe \n"
                    err_msg += "Choose one of: \"displacement\", \"velocity\", \"acceleration\", \"reaction\""
                    raise Exception(err_msg)
