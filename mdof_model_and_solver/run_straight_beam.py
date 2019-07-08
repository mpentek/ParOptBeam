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
Last update: 30.11.2017
'''
# ===============================================================================
import numpy as np
import matplotlib.pyplot as plt

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*

import json
import sys

parameter_file = open('ProjectParameters3DBeam.json', 'r')
parameters = json.loads(parameter_file.read())

beam_model = StraightBeam(parameters)

#==============================================
# #eigen value analysis
# eigenvalue_analysis = EigenvalueAnalysis(beam_model)
# eigenvalue_analysis.solve()
# eigenvalue_analysis.write_output_file()
# igenvalue_analysis.plot_selected_eigenmode(1)
# eigenvalue_analysis.plot_selected_eigenmode(4)
# eigenvalue_analysis.plot_selected_first_n_eigenmodes(3)
# eigenvalue_analysis.animate_selected_eigenmode(1)


#============================================
# #static analysis 
# static_force = 100000 * np.ones(120)
# print(static_force)
# static_analysis = StaticAnalysis(beam_model)
# static_analysis.solve(static_force)
# static_analysis.write_output_file() # TODO : write the function to write out outputs 
# static_analysis.plot_solve_result()

#===========================================
# Dynamic analysis 
# # time parameters
number_of_elements = parameters["model_parameters"]["system_parameters"]["geometry"]["number_of_elements"]
#  only applicable to fixed boundary condition 
force_data_folder = 'level_force'

array_time = np.loadtxt(force_data_folder+'/level_0.dat', usecols=(0,), skiprows=7)
dt = array_time[1] - array_time[0]
dynamic_force = np.zeros([len(beam_model.bcs_to_keep), len(array_time)])
num_dof = beam_model.DOFS_PER_NODE[beam_model.domain_size]
# extractign the forces from the files
for level in np.arange(number_of_elements):
        array_time_i = np.loadtxt(force_data_folder+'/level_'+str(level)+'.dat', usecols=(0,), skiprows=7)
        force_i = np.loadtxt(force_data_folder+'/level_'+str(level)+'.dat', usecols=(1,2,3,4,5,6), skiprows=7)
        if np.array_equal(array_time,array_time_i): 
                dynamic_force[level*num_dof:(level+1)*num_dof,:] = np.transpose(force_i)


# start_time = 0
# end_time = 10
# dt = 0.005
# array_time = np.arange (start_time,end_time + dt, dt)
print(dynamic_force)

# # # external dynamic force acting on the system
# freq = 1
# # # MDoFModel more degrees of freedom -> 2*len(Z) 2 degrees of freedom for each floor
# ampl_sin = 100000 * np.ones([120,1])
# dynamic_force = ampl_sin * np.sin ( 2 * np.pi * freq * array_time)
# print('dynamic forces', dynamic_force.shape)

# initial condition # TODO all the inital displacement and velocity are zeros . to incorporate non zeros values required ? 
dynamic_analysis = DynamicAnalysis(beam_model, dynamic_force, dt, array_time,
                        "GenAlpha" )
dynamic_analysis.solve()
dynamic_analysis.plot_selected_time_step(0.75)
dynamic_analysis.animate_time_history()
dynamic_analysis.plot_result_at_dof(30, 'acceleration')
dynamic_analysis.plot_result_at_dof(30, 'displacement')