# ===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18 
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
        Analysis type base class and derived classes specific types

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
Last update: 09.07.2019
'''
# ===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
import json

from source.structure_model import *
from source.time_integration_scheme import *

# import visualize_results_utilities to use it as a python object
from source import visualize_result_utilities


class AnalysisType(object):
    """
    Base class for the different analysis types
    """

    def __init__(self, structure_model, name="DefaultAnalysisType"):
        self.name = name

        # the structure model - geometry and physics - has the Dirichlet BC
        # for the bottom node included
        self.structure_model = structure_model

        self.displacement = None
        self.rotation = None

        self.force = None
        self.reaction = None
        self.moment = None

    def solve(self):
        """
        Solve for something
        """
        print("Solving for something in AnalysisType base class \n")
        pass


class StaticAnalysis(AnalysisType):
    """
    Dervied class for the static analysis of a given structure model        
    """

    def __init__(self, structure_model, name="StaticAnalysis"):

        super().__init__(structure_model, name)

    def solve(self, ext_force):
        print("Solving for ext_force in StaticAnalysis derived class \n")
        self.force = ext_force
        force = self.structure_model.apply_bc_by_reduction(self.force,'row_vector')
        

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)
        print(k)
        self.static_result = np.linalg.solve(k, force)    
        self.static_result = self.structure_model.recuperate_bc_by_extension(self.static_result, 'row_vector')
        self.force_action = {"x": np.zeros(0),
                        "y": np.zeros(0),
                        "z": np.zeros(0),
                        "a": np.zeros(0),
                        "b": np.zeros(0),
                        "g": np.zeros(0)}

        #self.force = self.structure_model.recuperate_bc_by_extension(self.force,'row_vector')
        self.resisting_force = self.force - np.matmul(self.structure_model.k,self.static_result)
        ixgrid = np.ix_(self.structure_model.bcs_to_keep, [0])
        self.resisting_force[ixgrid] = 0
        self.reaction = {"x": np.zeros(0),
                        "y": np.zeros(0),
                        "z": np.zeros(0),
                        "a": np.zeros(0),
                        "b": np.zeros(0),
                        "g": np.zeros(0)}
       
       
    def plot_solve_result(self):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting result in StaticAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.static_result.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.static_result[start:stop +
                                                                           1:step][:,0]
            self.force_action[label] = self.force[start:stop + 1:step]
            self.reaction[label] = self.resisting_force[start:stop + 1:step][:,0]

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
                            self.reaction["z"]] } 

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Static Analysis : "

        visualize_result_utilities.plot_result(plot_title,
                                               geometry,
                                               force,
                                               scaling,
                                               1)

    def write_output_file(self):
        """"
        This function writes out the nodal dofs of the deformed state

        """
        file = open("beam.txt", "w")
        dict = {}
        dict["length"] = max(self.structure_model.nodal_coordinates["x0"])
        dict["num_of_elements"] = len(self.structure_model.nodal_coordinates["x0"])
        for key, val in self.structure_model.nodal_coordinates.items():
            dict[key] = val.tolist()

        json_string = json.dumps(dict)

        file.write(json_string)
        file.close()


class EigenvalueAnalysis(AnalysisType):
    """
    Derived class for the (dynamic) eigenvalue analysis of a given structure model        
    """

    def __init__(self, structure_model, name="EigenvalueAnalysis"):
        super().__init__(structure_model, name)

        # adding additional attributes to the derived class
        self.eigenform = None
        self.frequency = None
        self.period = None

    def solve(self):

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)
        m = self.structure_model.apply_bc_by_reduction(self.structure_model.m)

        eig_values_raw, eig_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # s
        eig_pers = 1. / eig_freqs
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        ##

        # normalize results
        # http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect19.d/IAST.Lect19.Slides.pdf
        # normalize - unit generalized mass - slide 23

        [rows, columns] = eig_modes_raw.shape

        eig_modes_norm = np.zeros((rows, columns))

        gen_mass_raw = np.zeros(columns)
        gen_mass_norm = np.zeros(columns)

        print("Generalized mass should be identity")
        for i in range(len(eig_values_raw)):
            gen_mass_raw[i] = (np.transpose(eig_modes_raw[:, i])).dot(
                m).dot(eig_modes_raw[:, i])

            unit_gen_mass_norm_fact = np.sqrt(gen_mass_raw[i])

            eig_modes_norm[:, i] = eig_modes_raw[:, i]/unit_gen_mass_norm_fact

            gen_mass_norm[i] = (np.transpose(eig_modes_norm[:, i])).dot(
                m).dot(eig_modes_norm[:, i])
            # print("norm ", i, ": ",gen_mass_norm[i])

        # print("Multiplication check: thethaT dot M dot theta: ",(np.transpose(eig_modes_norm)).dot(self.structure_model.m).dot(eig_modes_norm)," numerically 0 for off-diagonal terms")
        # print()

        self.eigenform = np.zeros(eig_modes_norm.shape)
        self.frequency = np.zeros(eig_freqs.shape)
        self.period = np.zeros(eig_pers.shape)

        for index in range(len(eig_freqs)):
            self.eigenform[:, index] = eig_modes_norm[:,
                                                      eig_freqs_sorted_indices[index]]
            self.frequency[index] = eig_freqs[eig_freqs_sorted_indices[index]]
            self.period[index] = eig_pers[eig_freqs_sorted_indices[index]]

        self.eigenform = self.structure_model.recuperate_bc_by_extension(
            self.eigenform)

    def write_output_file(self):
        """"
        This function writes out the nodal dofs of the deformed state

        """
        file = open("beam.txt", "w")
        dict = {}
        dict["length"] = max(self.structure_model.nodal_coordinates["x0"])
        dict["num_of_elements"] = len(self.structure_model.nodal_coordinates["x0"])
        for key, val in self.structure_model.nodal_coordinates.items():
            dict[key] = val.tolist()

        json_string = json.dumps(dict)

        file.write(json_string)
        file.close()

    def plot_selected_eigenmode(self, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend

        """
        selected_mode = selected_mode - 1

        print("Plotting result for a selected eigenmode in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode]

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

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

        plot_title = " Eigenmode: " + str(selected_mode+1)
        plot_title += "  Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode])
        plot_title += "  Period: " + \
            '{0:.2f}'.format(self.period[selected_mode])

        visualize_result_utilities.plot_result(plot_title,
                                               geometry,
                                               force,
                                               scaling,
                                               1)

    def plot_selected_first_n_eigenmodes(self, number_of_modes):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend
        """

        print("Plotting result for selected first n eigenmodes in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop+1:step]

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

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

        plot_title = " "
        for selected_mode in range(number_of_modes):
            plot_title += "Eigenmode " + str(selected_mode + 1) + "  Frequency: " + str(np.round(
                self.frequency[selected_mode], 3)) + "  Period: " + str(np.round(self.period[selected_mode], 3)) + "\n"
        visualize_result_utilities.plot_result(plot_title,
                                               geometry,
                                               force,
                                               scaling,
                                               number_of_modes)

    def animate_selected_eigenmode(self, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend  
        """
        selected_mode = selected_mode - 1

        print("Animating eigenmode in EigenvalueAnalysis \n")

        time_steps = 100
        array_time = np.sin(2 * np.pi * self.frequency[selected_mode] * np.linspace(
            0, self.period[selected_mode], time_steps))  # AK: can this be called an array tiem ? 

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode][:, np.newaxis] * array_time

        # for remaining dofs - case of 2D - create placeholders in correct format
        remaining_labels = list(set(StraightBeam.DOF_LABELS['3D'])-set(
            StraightBeam.DOF_LABELS[self.structure_model.domain_size]))
        for label in remaining_labels:
            self.structure_model.nodal_coordinates[label] = np.zeros(
                (self.structure_model.n_nodes, len(array_time)))

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

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

        plot_title = "Eigenmode: " + str(selected_mode)
        plot_title += "  Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode])
        plot_title += "  Period: " + \
            '{0:.2f}'.format(self.period[selected_mode])

        visualize_result_utilities.animate_result(plot_title,
                                                  array_time,
                                                  geometry,
                                                  force,
                                                  scaling)


class DynamicAnalysis(AnalysisType):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    def __init__(self, structure_model, force, dt, array_time, time_integration_scheme="GenAlpha", name="DynamicAnalysis"):
        super().__init__(structure_model, name)
        # print("Force: ", len(force))
        # overwriting attribute from base constructors
        self.force = force
        self.dt = dt 
        #self.time = time
        self.array_time = np.asarray(array_time) #np.arange(self.time[0], self.time[1] + self.dt, self.dt)
        rows = len(self.structure_model.apply_bc_by_reduction(self.structure_model.k))
        cols = len(self.array_time)

        # inital condition of zero displacement and velocity used for the time being. 
        # TODO : to incorporate user defined initial conditions

        u0 = np.zeros(rows) # initial displacement
        v0 = np.zeros(rows)  # initial velocity
        a0 = np.zeros(rows)  # initial acceleration
        initial_conditions = np.array([u0,v0,a0])


        # adding additional attributes to the derived class
        self.displacement = np.zeros((rows, cols))
        self.velocity = np.zeros((rows, cols))
        self.acceleration = np.zeros((rows, cols))

        if force.shape[1] != len(self.array_time):
            err_msg = "The time step for forces does not match the time step defined"
            raise Exception(err_msg)

        if time_integration_scheme == "GenAlpha":
            self.solver = GeneralizedAlphaScheme(
                self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "Euler12":
            self.solver = Euler12(self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "ForwardEuler1":
            self.solver = ForwardEuler1(
                self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "BackwardEuler1":
            self.solver = BackwardEuler1(
                self.dt, structure_model, initial_conditions)

        else:
            err_msg = "The requested time integration scheme \"" + time_integration_scheme
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"GenAlpha\", \"Euler12\", \"ForwardEuler1\", \"BackwardEuler1\""
            raise Exception(err_msg)

    def solve(self):
        print("Solving the structure for dynamic loads \n")
        force = self.structure_model.apply_bc_by_reduction(self.force,'row')
        # time loop
        for i in range(1, len(self.array_time)):
            current_time = self.array_time[i]
            self.solver.solve_structure(force[:, i]-force[:, i-1])

            # appending results to the list
            self.displacement[:, i] = self.solver.get_displacement()
            self.velocity[:, i] = self.solver.get_velocity()
            self.acceleration[:, i] = self.solver.get_acceleration()

            # update results
            self.solver.update_structure_time_step()
        
        self.displacement = self.structure_model.recuperate_bc_by_extension(self.displacement)
        self.velocity = self.structure_model.recuperate_bc_by_extension(self.velocity)
        self.acceleration = self.structure_model.recuperate_bc_by_extension(self.acceleration)

    def compute_reactions(self):
        # forward multiplying to compute the forces and reactions 
        self.dynamic_reaction = self.force - np.matmul(self.structure_model.m,self.displacement)
        -np.matmul(self.structure_model.b, self.velocity)
        -np.matmul(self.structure_model.k, self.displacement)
        
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

        visualize_result_utilities.plot_dynamic_result(plot_title, result_data, self.array_time)


    def plot_selected_time_step(self, selected_time_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  

        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")

        # find closesed time step
        idx_time = (np.abs(self.array_time-selected_time_step)).argmin()

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

        force = {"external": None ,# [np.append(origin_point, self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": None } #[np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Dyanimc Analyis: Deformation at t = " + \
            str(selected_time_step) + " [s]"

        visualize_result_utilities.plot_result(plot_title,
                                               geometry,
                                               force,
                                               scaling,
                                               1)

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

        visualize_result_utilities.animate_result(plot_title,
                                                  self.array_time,
                                                  geometry,
                                                  force,
                                                  scaling)
