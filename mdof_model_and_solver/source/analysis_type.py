#===============================================================================
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
Last update: 30.11.2017
'''
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys

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

        print(self.structure_model.category)
        
        self.result = np.linalg.solve(self.structure_model.k, self.force)

        if self.structure_model.category in ['SDoF','MDoFShear']:
            self.displacement = self.result
            self.reaction = -1/2 * self.structure_model.k[0,0] * self.displacement[0] 
           
        elif self.structure_model.category in ['MDoFBeam', 'MDoFMixed']:
            self.displacement = self.result[::2]
            self.rotation = self.result[1::2]

            self.reaction = -1/2 * self.structure_model.k[0,0] * self.displacement[0] 
            self.moment = -1/2 * self.structure_model.k[0,1] * self.rotation[0] 

            self.reaction = -1/2 * self.structure_model.k[0,0] * self.displacement[0] 
            self.moment = -1/2 * self.structure_model.k[0,1] * self.rotation[0]

        else:
            sys.exit('Selected structure not implemented!')

        self.structure_model.nodal_coordinates["x"] = np.add(self.structure_model.nodal_coordinates["x0"], self.displacement)
        self.structure_model.nodal_coordinates["y"] = self.structure_model.nodal_coordinates["y0"]

    def plot_solve_result(self):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting result in StaticAnalysis \n")

        origin_point = np.zeros(1)

        geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
                                    np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
                    "deformation": [np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["x"]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["x0"])),
                                    np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))],
                    "deformed": None}

        force = {"external"     : [np.append(origin_point,self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}
        
        plot_title = "Static Analysis : "      

        visualize_result_utilities.plot_result(plot_title, 
                                               geometry,
                                               force,
                                               scaling, 
                                               1)


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
        eig_freqs = eig_values /2. /np.pi 
        # s
        eig_pers = 1. / eig_freqs   
        #sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)  
        ##

        ## normalize results
        # http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect19.d/IAST.Lect19.Slides.pdf
        # normalize - unit generalized mass - slide 23

        [rows, columns] = eig_modes_raw.shape
        
        eig_modes_norm = np.zeros((rows, columns))

        gen_mass_raw = np.zeros(columns)
        gen_mass_norm = np.zeros(columns)
        
        print("Generalized mass should be identity")
        for i in range(len(eig_values_raw)):
            gen_mass_raw[i] = (np.transpose(eig_modes_raw[:,i])).dot(m).dot(eig_modes_raw[:,i])

            unit_gen_mass_norm_fact = np.sqrt(gen_mass_raw[i])

            eig_modes_norm[:,i] = eig_modes_raw[:,i]/unit_gen_mass_norm_fact

            gen_mass_norm[i] = (np.transpose(eig_modes_norm[:,i])).dot(m).dot(eig_modes_norm[:,i])
            # print("norm ", i, ": ",gen_mass_norm[i])

        # print("Multiplication check: thethaT dot M dot theta: ",(np.transpose(eig_modes_norm)).dot(self.structure_model.m).dot(eig_modes_norm)," numerically 0 for off-diagonal terms")
        # print()

        self.eigenform = np.zeros(eig_modes_norm.shape)  
        self.frequency = np.zeros(eig_freqs.shape)
        self.period = np.zeros(eig_pers.shape) 

        for index in range(len(eig_freqs)):
            self.eigenform[:, index] = eig_modes_norm[:, eig_freqs_sorted_indices[index]]
            self.frequency[index] =  eig_freqs[eig_freqs_sorted_indices[index]]
            self.period[index] = eig_pers[eig_freqs_sorted_indices[index]]

        self.eigenform = self.structure_model.recuperate_bc_by_extension(self.eigenform)

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

        if self.structure_model.category in ['SDoF','MDoFShear']:
            self.structure_model.nodal_coordinates["x"] = np.add(self.structure_model.nodal_coordinates["x0"], self.eigenform[:,selected_mode])

        elif self.structure_model.category in ['MDoFBeam','MDoFMixed']:
            self.structure_model.nodal_coordinates["x"] = np.add(self.structure_model.nodal_coordinates["x0"], self.eigenform[::2,selected_mode])
            
        else:
            sys.exit('Wrong structural type selected for eigenmodes')
           
        self.structure_model.nodal_coordinates["y"] = self.structure_model.nodal_coordinates["y0"]
        origin_point = np.zeros(1)

        # geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
        #                             np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
        #             "deformation": [np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["x"]),
        #                                         np.append(origin_point, self.structure_model.nodal_coordinates["x0"])),
        #                             np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"]),
        #                                         np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))],
        #             "deformed": None}


        geometry = {"undeformed" : [self.structure_model.nodal_coordinates["x0"],
                                    self.structure_model.nodal_coordinates["y0"]],
                    "deformation": [np.subtract(self.structure_model.nodal_coordinates["x"],
                                                self.structure_model.nodal_coordinates["x0"]),
                                    np.subtract(self.structure_model.nodal_coordinates["y"],
                                                self.structure_model.nodal_coordinates["y0"])],
                    "deformed": None}
                    
        force = {"external"     : None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}
          
        plot_title = " Eigenmode: " + str(selected_mode+1) 
        plot_title += "  Frequency: " + '{0:.2f}'.format(self.frequency[selected_mode]) 
        plot_title += "  Period: " + '{0:.2f}'.format(self.period[selected_mode])

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

        if self.structure_model.category in ['SDoF','MDoFShear']:
            self.structure_model.nodal_coordinates["x"] = self.eigenform

        elif self.structure_model.category in ['MDoFBeam','MDoF2DMixed']:
            self.structure_model.nodal_coordinates["x"] = self.eigenform[::2]

        elif self.structure_model.category in ['MDoF3DMixed']:
            # TODO: make this generic
            # x, y, z, alpha, beta, gamma
            # slicing: start:stop:step
            chosen_dof = 1 # should be y or z
            dofs_per_node = 6
            start = chosen_dof
            stop = self.eigenform.shape[0] + chosen_dof - dofs_per_node
            step = dofs_per_node
            self.structure_model.nodal_coordinates["x"] = self.eigenform[start:stop+1:step]
            
            #self.structure_model.nodal_coordinates["x"] = self.eigenform[::6]

        else:
            sys.exit("If this error message appears, there is a bug in the code, please contact your supervisor")

        self.structure_model.nodal_coordinates["y"] = self.structure_model.nodal_coordinates["y0"]

        # origin_point = np.zeros(1)
        # origin_vector = np.zeros(len(self.eigenform))

        # geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
        #                             np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
        #             "deformation": [np.vstack((origin_vector, self.structure_model.nodal_coordinates["x"])),
        #                             np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"]),
        #                                         np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))],
        #             "deformed": None}

        # NOTE: bc dofs should already be recoperated
        # TODO: check if an origin point shift or extension still needed
        geometry = {"undeformed" : [self.structure_model.nodal_coordinates["x0"],
                                    self.structure_model.nodal_coordinates["y0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    np.subtract(self.structure_model.nodal_coordinates["y"],
                                                self.structure_model.nodal_coordinates["y0"])],
                    "deformed": None}

        force = {"external"     : None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}
        # print("Geometry: ", geometry)
        # print("Self.Nodal coordinates: ", self.structure_model.nodal_coordinates["x"])
       
        plot_title = " "
        for selected_mode in range(number_of_modes):
            plot_title += "Eigenmode " + str(selected_mode +1) + "  Frequency: " + str(np.round(self.frequency[selected_mode],3)) + "  Period: " + str(np.round(self.period[selected_mode],3)) + "\n"       
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

        if self.structure_model.category in ['SDoF','MDoFShear']:
            displacement = self.eigenform[:, selected_mode]

        elif self.structure_model.category in ['MDoFBeam', 'MDoFMixed']:
            displacement = self.eigenform[::2, selected_mode]

        else:
            sys.exit("If this error message appears, there is a bug in the code, please contact your supervisor")
    
       # displacement = self.eigenform[:, selected_mode]
        time_steps = 100
        array_time = np.linspace(0, self.period[selected_mode], time_steps)

        displacement_time_history = [[] for i in range(len(array_time))]
        for i in range(len(array_time)):
            displacement_time_history[i] = [value * np.sin(
                2 * np.pi * self.frequency[selected_mode] * array_time[i]) for value in displacement]
          
        self.structure_model.nodal_coordinates["x"] = [[] for i in range(len(array_time))]
        self.structure_model.nodal_coordinates["y"] = [[] for i in range(len(array_time))]
        for i in range(len(array_time)):
            self.structure_model.nodal_coordinates["x"][i] = displacement_time_history[i]
            self.structure_model.nodal_coordinates["y"][i] = self.structure_model.nodal_coordinates["y0"]

        origin_point = np.zeros(1)

        geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
                                    np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
                    "deformation": [[[] for i in range(len(array_time))], [[] for i in range(len(array_time))]],
                    "deformed": [[[] for i in range(len(array_time))], [[] for i in range(len(array_time))]]}

        for i in range(len(array_time)):
           
            geometry["deformation"][0][i] =np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["x"][i]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["x0"]))
            geometry["deformation"][1][i] =np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"][i]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))

        force = {"external"     : None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}
        
        plot_title = "Eigenmode: " + str(selected_mode) 
        plot_title += "  Frequency: " + '{0:.2f}'.format(self.frequency[selected_mode]) 
        plot_title += "  Period: " + '{0:.2f}'.format(self.period[selected_mode])

        visualize_result_utilities.animate_result(plot_title, 
                                                 array_time,
                                                 geometry,
                                                 force,
                                                 scaling)


class DynamicAnalysis(AnalysisType):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    def __init__(self, structure_model, initial_conditions, force, time, dt, time_integration_scheme="GenAlpha", name="DynamicAnalysis"):

        super().__init__(structure_model, name)
        # print("Force: ", len(force))
        # overwriting attribute from base constructors
        self.force = force

        self.dt = dt
        self.time = time
        self.array_time = np.arange(self.time[0], self.time[1] + self.dt, self.dt)
        rows = len(self.force)
        cols = len(self.array_time)

        # adding additional attributes to the derived class
        self.displacement = np.zeros((rows, cols)) 
        self.velocity = np.zeros((rows, cols))
        self.acceleration = np.zeros((rows, cols)) 

        if self.force.shape[1] != len(self.array_time):
            err_msg =  "The time step for forces does not match the time step defined"
            raise Exception(err_msg)
       
        if time_integration_scheme == "GenAlpha":
            self.solver = GeneralizedAlphaScheme(self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "Euler12":
            self.solver = Euler12(self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "ForwardEuler1":
            self.solver = ForwardEuler1(self.dt, structure_model, initial_conditions)
        elif time_integration_scheme == "BackwardEuler1":
            self.solver = BackwardEuler1(self.dt, structure_model, initial_conditions)
            
        else:
            err_msg =  "The requested time integration scheme \"" + time_integration_scheme 
            err_msg +=  "\" is not available \n"
            err_msg += "Choose one of: \"GenAlpha\", \"Euler12\", \"ForwardEuler1\", \"BackwardEuler1\""
            raise Exception(err_msg)

    def solve(self):
        print("Solving the structure for dynamic loads \n")
        # time loop 
        for i in range(1,len(self.array_time)): 
            current_time = self.array_time[i]
            self.solver.solve_structure(self.force[:,i])

            # appending results to the list 
            self.displacement[:,i] = self.solver.get_displacement()
            self.velocity[:,i] = self.solver.get_velocity()
            self.acceleration[:,i] = self.solver.get_acceleration()

            # update results
            self.solver.update_structure_time_step()   
        
 
    def plot_selected_time_step(self, selected_time_step):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacement -> here as time series -> select closes results to a requested time_step [s]  
            
        """

        print("Plotting result for a selected time step in DynamicAnalysis \n")
                
        # find closesed time step 
        idx = (np.abs(self.array_time-selected_time_step)).argmin()

         # TODO for the dynamic analysis create self.structure_model.nodal_coordinates after solve not here
        if self.structure_model.category in ['SDoF','MDoFShear']:
            self.structure_model.nodal_coordinates["x"] = self.displacement[:,idx]
        elif self.structure_model.category in ['MDoFBeam', 'MDoFMixed']:
            self.structure_model.nodal_coordinates["x"] = self.displacement[::2,idx]
        else:
            sys.exit()

        self.structure_model.nodal_coordinates["y"] = self.structure_model.nodal_coordinates["y0"]

        origin_point = np.zeros(1)

        geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
                                    np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
                    "deformation": [np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["x"]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["x0"])),
                                    np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))],
                    "deformed": None}

        force = {"external"     : [np.append(origin_point,self.force), np.zeros(len(self.force) + 1)],
                 "base_reaction": [np.append(self.reaction, origin_point), np.zeros(len(self.force) + 1)]}

        scaling = {"deformation": 1,
                   "force": 1}
        
        plot_title = "Dyanimc Analyis: Deformation at t = " + str(selected_time_step) + " [s]"       

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
          
        self.structure_model.nodal_coordinates["x"] = [[] for i in range(len(self.array_time))]
        self.structure_model.nodal_coordinates["y"] = [[] for i in range(len(self.array_time))]
        for i in range(len(self.array_time)):
            if self.structure_model.category in ['SDoF','MDoFShear']:
                self.structure_model.nodal_coordinates["x"][i] = self.displacement[:,i]
            elif self.structure_model.category in ['MDoFBeam', 'MDoFMixed']:
                self.structure_model.nodal_coordinates["x"][i] = self.displacement[::2,i]
            else:
                sys.exit()

            self.structure_model.nodal_coordinates["y"][i] = self.structure_model.nodal_coordinates["y0"]

        origin_point = np.zeros(1)

        geometry = {"undeformed" : [np.append(origin_point, self.structure_model.nodal_coordinates["x0"]),
                                    np.append(origin_point, self.structure_model.nodal_coordinates["y0"])],
                    "deformation": [[[] for i in range(len(self.array_time))], [[] for i in range(len(self.array_time))]],
                    "deformed": [[[] for i in range(len(self.array_time))], [[] for i in range(len(self.array_time))]]}

        for i in range(len(self.array_time)):
            geometry["deformation"][0][i] =np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["x"][i]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["x0"]))
            geometry["deformation"][1][i] =np.subtract(np.append(origin_point, self.structure_model.nodal_coordinates["y"][i]),
                                                np.append(origin_point, self.structure_model.nodal_coordinates["y0"]))

        force = {"external"     : None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}
        
        plot_title = "Dyanimc Analyis: Deformation over time"   

        visualize_result_utilities.animate_result(plot_title, 
                                                  self.array_time,
                                                  geometry,
                                                  force,
                                                  scaling)
