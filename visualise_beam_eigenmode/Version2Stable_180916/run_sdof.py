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
Last update: 23.11.2017
'''
#===============================================================================

import numpy as np

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*


# =========================sdof system properties==========================  
mass = 1 # the mass of the sdof system
target_freq = 1 # the target frequency of the sdof system for which stiffness may be tuned
zeta = 0.05 # the damping ratio of the sdof system

# model definition 
sdof_model = SDoFModel(mass, target_freq, zeta)

#========================= static analysis==========================  
force_static = np.array([1.5]) # external static force acting on the system

static_analysis = StaticAnalysis(sdof_model)
static_analysis.solve(force_static)
static_analysis.plot_solve_result()

#========================= eigenvalue analysis ==========================  
eigenvalue_analysis = EigenvalueAnalysis(sdof_model)
eigenvalue_analysis.solve()
eigenvalue_analysis.plot_selected_eigenmode(1)
eigenvalue_analysis.animate_selected_eigenmode(1)

#========================= dynamic analysis ==========================  

# time parameters 
start_time = 0
end_time = 10
dt = 0.01
array_time = np.arange (start_time,end_time + dt, dt)

# dynamic forces
"""
Choose from "signalSin", "signalRand", "signalConst", "signalSuperposed" or 
for free vibration choose "signalNone" 
"""
# external dynamic force acting on the system
freq = 10
force_dynamic = load_type("signalSin", array_time, 1, freq, force_static) 

# inital condition 

u0 = [0.] # initial displacement
v0 = [0.]  # initial velocity
a0 = [0.]  # initial acceleration

"""
Numerical Integration type : Choose from "GenAlpha", "Euler12", "ForwardEuler1", "BackwardEuler1"
"""

dynamic_analysis = DynamicAnalysis(sdof_model, np.array([u0,v0,a0]), force_dynamic, [start_time,end_time], 
                   dt, "GenAlpha" ) 

dynamic_analysis.solve()
dynamic_analysis.plot_selected_time_step(0.75)
dynamic_analysis.animate_time_history()
