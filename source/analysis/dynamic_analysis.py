import numpy as np
import json
from os.path import join, isdir
from os import makedirs

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.writer_utilitites as writer_utilities
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults


class DynamicAnalysis(AnalysisType):
    """
    Dervied class for the dynamic analysis of a given structure model        

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
        steps = int((self.parameters['settings']['time']['end']-self.parameters['settings']
                     ['time']['start'])/self.dt) + 1
        self.array_time = np.linspace(start, stop, steps)

        # load parameters
        '''
        FOR NOW ONLY AVAILABLE for
        1 elements - 2 nodes
        2 elements - 3 nodes
        3 elements - 4 nodes
        6 elements - 7 nodes
        12 elements - 13 nodes
        24 elements - 25 nodes
        '''
        possible_n_el_cases = [1, 2, 3, 6, 12, 24]
        if structure_model.parameters['n_el'] not in possible_n_el_cases:
            err_msg = "The number of element input \"" + \
                str(structure_model.parameters['n_el'])
            err_msg += "\" is not allowed for Dynamic Analysis \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join([str(x) for x in possible_n_el_cases])
            raise Exception(err_msg)
        # TODO include some specifiers in the parameters, do not hard code
        force = np.load(join(*['input', 'force', 'force_dynamic' +
                               '_turb' + str(structure_model.parameters['n_el']+1) + '.npy']))

        super().__init__(structure_model, self.parameters["type"])
        # print("Force: ", len(force))
        # overwriting attribute from base constructors
        self.force = force

        #self.time = time
        # np.arange(self.time[0], self.time[1] + self.dt, self.dt)

        rows = len(self.structure_model.apply_bc_by_reduction(
            self.structure_model.k))
        cols = len(self.array_time)

        # inital condition of zero displacement and velocity used for the time being.
        # TODO : to incorporate user defined initial conditions

        u0 = np.zeros(rows)  # initial displacement
        v0 = np.zeros(rows)  # initial velocity
        a0 = np.zeros(rows)  # initial acceleration
        initial_conditions = np.array([u0, v0, a0])

        # adding additional attributes to the derived class
        self.displacement = np.zeros((rows, cols))
        self.velocity = np.zeros((rows, cols))
        self.acceleration = np.zeros((rows, cols))

        if force.shape[1] != len(self.array_time):
            err_msg = "The time step for forces does not match the time step defined"
            raise Exception(err_msg)

        if time_integration_scheme == "GenAlpha":
            from source.scheme.generalized_alpha_scheme import GeneralizedAlphaScheme
            self.solver = GeneralizedAlphaScheme(
                self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "Euler12":
            from source.scheme.euler12_scheme import Euler12
            self.solver = Euler12(self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "ForwardEuler1":
            from source.scheme.forward_euler1_scheme import ForwardEuler1
            self.solver = ForwardEuler1(
                self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "BackwardEuler1":
            from source.scheme.backward_euler1_scheme import BackwardEuler1
            self.solver = BackwardEuler1(
                self.dt, structure_model, initial_conditions)

        else:
            err_msg = "The requested time integration scheme \"" + time_integration_scheme
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"GenAlpha\", \"Euler12\", \"ForwardEuler1\", \"BackwardEuler1\""
            raise Exception(err_msg)

    def solve(self):
        print("Solving the structure for dynamic loads \n")
        force = self.structure_model.apply_bc_by_reduction(self.force, 'row')
        # time loop
        for i in range(1, len(self.array_time)):
            current_time = self.array_time[i]

            self.solver.solve_structure(force[:, i])

            # appending results to the list
            self.displacement[:, i] = self.solver.get_displacement()
            self.velocity[:, i] = self.solver.get_velocity()
            self.acceleration[:, i] = self.solver.get_acceleration()

            # update results
            self.solver.update_structure_time_step()

        self.displacement = self.structure_model.recuperate_bc_by_extension(
            self.displacement)
        self.velocity = self.structure_model.recuperate_bc_by_extension(
            self.velocity)
        self.acceleration = self.structure_model.recuperate_bc_by_extension(
            self.acceleration)

    def compute_reactions(self):
        # forward multiplying to compute the forces and reactions
        # NOTE: check if this is needed, seems to be unused
        # ixgrid = np.ix_(self.structure_model.bcs_to_keep, [0])

        f1 = np.matmul(self.structure_model.m, self.acceleration)
        f2 = np.matmul(self.structure_model.b, self.velocity)
        f3 = np.matmul(self.structure_model.k, self.displacement)
        self.dynamic_reaction = self.force - f1 - f2 - f3

        # TODO: check if the treatment of elastic bc dofs is correct
        for dof_id, stiffness_val in self.structure_model.elastic_bc_dofs.items():
            # assuming a Rayleigh-model
            damping_val = stiffness_val * self.structure_model.a[1]

            f1 = 0.0 * self.acceleration[dof_id]
            f2 = damping_val * self.velocity[dof_id]
            f3 = stiffness_val * self.displacement[dof_id]

            # overwrite the existing value with one solely from spring stiffness and damping
            self.dynamic_reaction[dof_id] = f1 + f2 + f3

    def write_result_at_dof(self, dof, selected_result):
        """"
        This function writes out the time history of response at the selected dof

        """
        print('Writing out result for selected dof in dynamic analysis \n')
        if selected_result == 'displacement':
            result_data = self.displacement[dof, :]
        elif selected_result == 'velocity':
            result_data = self.velocity[dof, :]
        elif selected_result == 'acceleration':
            result_data = self.acceleration[dof, :]
        else:
            err_msg = "The selected result \"" + selected_result
            err_msg += "\" is not avaialbe \n"
            err_msg += "Choose one of: \"displacement\", \"velocity\", \"acceleration\""
            raise Exception(err_msg)

        file = open(selected_result + "at_dof_" + str(dof) + ".txt", "w")
        file.write("time \t" + selected_result + "\n")
        for i in np.arange(len(self.array_time)):
            file.write(str(self.array_time[i])+"\t"+str(result_data[i])+"\n")
        file.close()

    def plot_result_at_dof(self, dof, selected_result):
        """
        Pass to plot function:
            Plots the time series of required quantitiy 
        """
        print('Plotting result for selected dof in dynamic analysis \n')
        plot_title = selected_result.capitalize() + ' at DoF ' + str(dof)
        if selected_result == 'displacement':
            result_data = self.displacement[dof, :]
        elif selected_result == 'velocity':
            result_data = self.velocity[dof, :]
        elif selected_result == 'acceleration':
            result_data = self.acceleration[dof, :]
        else:
            err_msg = "The selected result \"" + selected_result
            err_msg += "\" is not avaialbe \n"
            err_msg += "Choose one of: \"displacement\", \"velocity\", \"acceleration\""
            raise Exception(err_msg)

        plotter_utilities.plot_dynamic_result(
            plot_title, result_data, self.array_time)

    def plot_reaction(self, dof):
        self.compute_reactions()
        """
        Pass to plot function:
            Plots the time series of required quantitiy 
        """
        print('Plotting reactions for in dynamic analysis \n')
        # TODO: check if the treatment of elastic bc dofs is correct
        if dof in self.structure_model.bc_dofs or dof in self.structure_model.elastic_bc_dofs:
            plot_title = 'REACTION at DoF ' + str(dof)
            plotter_utilities.plot_dynamic_result(
                plot_title, self.dynamic_reaction[dof, :], self.array_time)
        else:
            err_msg = "The selected DoF \"" + str(dof)
            err_msg += "\" is not avaialbe in the list of available boundary condition dofs \n"
            err_msg += "Choose one of: " + \
                ", ".join([str(val) for val in self.structure_model.bc_dofs])
            raise Exception(err_msg)

    def plot_selected_time(self, pdf_report, display_plots, selected_time):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")

        # find closesed time step
        idx_time = np.where(self.array_time >= selected_time)[0][0]

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.displacement[start:stop +
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

    def write_selected_time(self, selected_time):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")

        # find closesed time step
        idx_time = np.where(self.array_time >= selected_time)[0][0]

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.displacement[start:stop +
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
        absolute_folder_path = join(
            "output", self.structure_model.name)
        # make sure that the absolute path to the desired output folder exists
        if not isdir(absolute_folder_path):
            makedirs(absolute_folder_path)

        writer_utilities.write_result(join(absolute_folder_path, file_name), file_header,
                                      geometry, scaling)

    def plot_selected_step(self, pdf_report, display_plots, selected_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected step in DynamicAnalysis \n")

        # TODO refactor so that plot_selected_time calls plot_selected_step
        idx_time = selected_step

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.displacement[start:stop +
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

        plot_title = "Dyanimc Analyis: Deformation for step = " + str(idx_time) + " at t = " + \
            str(self.array_time[idx_time]) + " [s]"

        plotter_utilities.plot_result(pdf_report,
                                      display_plots,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      1)

    def write_selected_step(self, selected_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected step in DynamicAnalysis \n")

        # TODO refactor so that plot_selected_time calls plot_selected_step
        idx_time = selected_step

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.displacement[start:stop +
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

        file_header = "# Dyanimc Analyis: Deformation for step = " + str(idx_time) + " at t = " + \
            str(self.array_time[idx_time]) + " [s]"

        file_name = 'dynamic_analysis_selected_step_' + str(idx_time) + '.dat'
        absolute_folder_path = join(
            "output", self.structure_model.name)
        # make sure that the absolute path to the desired output folder exists
        if not isdir(absolute_folder_path):
            makedirs(absolute_folder_path)

        writer_utilities.write_result(join(absolute_folder_path, file_name), file_header,
                                      geometry, scaling)

    def animate_time_history(self):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series  
        """

        print("Animating time history in DynamicAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.displacement.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.displacement[start:stop +
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

    def postprocess(self, pdf_report, display_plots):
        """
        Postprocess something
        """
        print("Postprocessing in DynamicAnalysis derived class \n")

        for time in self.parameters['output']['selected_instance']['plot_time']:
            self.plot_selected_time(pdf_report, display_plots, time)

        for time in self.parameters['output']['selected_instance']['write_time']:
            self.write_selected_time(time)
            pass

        for step in self.parameters['output']['selected_instance']['plot_step']:
            self.plot_selected_step(pdf_report, display_plots, step)
            pass

        for step in self.parameters['output']['selected_instance']['write_step']:
            self.write_selected_step(step)
            pass

        if self.parameters['output']['animate_time_history']:
            self.animate_time_history()

        for idx_dof, dof_id in enumerate(self.parameters['output']['selected_dof']['dof_list']):

            # TODO unify for reaction and kinematics results

            for idx_res, res in enumerate(self.parameters['output']['selected_dof']['result_type'][idx_dof]):
                if res == 'reaction':
                    if self.parameters['output']['selected_dof']['plot_result'][idx_dof][idx_res]:
                        self.plot_reaction(dof_id)
                    if self.parameters['output']['selected_dof']['write_result'][idx_dof][idx_res]:
                        # self.write_reaction(dof_id)
                        pass
                elif res in ['displacement', 'velocity', 'acceleration']:
                    if self.parameters['output']['selected_dof']['plot_result'][idx_dof][idx_res]:
                        self.plot_result_at_dof(dof_id, res)
                    if self.parameters['output']['selected_dof']['write_result'][idx_dof][idx_res]:
                        # self.write_result_at_dof(dof_id, res)
                        pass
                else:
                    pass
        pass
