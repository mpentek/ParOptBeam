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

parameter_file = open('ProjectParameters3DBeam.json', 'r')
parameters = json.loads(parameter_file.read())

beam_model = StraightBeam(parameters)

eigenvalue_analysis = EigenvalueAnalysis(beam_model)
eigenvalue_analysis.solve()
#eigenvalue_analysis.write_output_file()
#eigenvalue_analysis.plot_selected_eigenmode(1)
#eigenvalue_analysis.plot_selected_eigenmode(4)
#eigenvalue_analysis.plot_selected_first_n_eigenmodes(3)
#eigenvalue_analysis.animate_selected_eigenmode(1)



# static analysis
static_force = 10 * np.ones(120)
print(static_force)
static_analysis = StaticAnalysis(beam_model)
static_analysis.solve(static_force)
#static_analysis.write_output_file()
#static_analysis.plot_solve_result()

# Dynamic analysis 

# # time parameters
start_time = 0
end_time = 10
dt = 0.005
array_time = np.arange (start_time,end_time + dt, dt)

# # external dynamic force acting on the system
freq = 1

# # MDoFModel more degrees of freedom -> 2*len(Z) 2 degrees of freedom for each floor
ampl_sin = 1000 * np.ones([120,1])
dynamic_force = ampl_sin * np.sin ( 2 * np.pi * freq * array_time)
print('dynamic forces', dynamic_force)
# initial condition # TODO : check a abetter way to do this 
u0 = np.zeros(120) # initial displacement
v0 = np.zeros(120)  # initial velocity
a0 = np.zeros(120)  # initial acceleration

dynamic_analysis = DynamicAnalysis(beam_model, np.array([u0,v0,a0]), dynamic_force, [start_time,end_time],
                    dt, "GenAlpha" )

dynamic_analysis.solve()
dynamic_analysis.plot_selected_time_step(0.75)
dynamic_analysis.animate_time_history()