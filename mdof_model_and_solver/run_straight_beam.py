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
import os

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*

import json


# ==============================================
# Model choice


'''
TODO: check model parameters for correctness
'''

'''
Pylon model with the extracted geometry from the CAD model
and matching structural to the solid models used for one-way coupling
'''
# parameter_file = open('ProjectParameters3DPylonCadBeam.json', 'r')

''' 
Pylon model with geometry data from the sofistik beam
material parameters also from sofistik sheet 
'''
# parameter_file = open('ProjectParameters3DPylonSofiBeam.json', 'r')
# parameter_file = open('ProjectParameters3DPylonSofiBeamReducedE.json', 'r')

# parameter_file = open('ProjectParameters3DPylonSofiBeamWithFoundationSoft.json', 'r')
# parameter_file = open('ProjectParameters3DPylonSofiBeamWithFoundationMid.json', 'r')
# parameter_file = open('ProjectParameters3DPylonSofiBeamWithFoundationHard.json', 'r')

'''
Equivalent beam model of the CAARC building B
assuming continous mass and stiffness (material and geometric)
distribution along the heigth (length of the beam in local coordinates)
'''
#parameter_file = open('ProjectParameters3DCaarcBeam.json', 'r')

'''
A prototype alternative to the CAARC building B with 3 intervals
to define altering geometric properties
'''
parameter_file = open('ProjectParameters3DCaarcBeamPrototype.json', 'r')


# ==============================================
# Parameter read


parameters = json.loads(parameter_file.read())
beam_model = StraightBeam(parameters)

beam_model.identify_decoupled_eigenmodes()

#beam_model.plot_model_properties()

# ==============================================
# Eigenvalue analysis


'''
TODO: check eigenvalue analysis with number of elements
3, 6, 12, 24, 48, 96
'''

eigenvalue_analysis = EigenvalueAnalysis(beam_model)
eigenvalue_analysis.solve()

# eigenvalue_analysis.write_output_file()

eigenvalue_analysis.plot_selected_eigenmode(1)
# eigenvalue_analysis.plot_selected_eigenmode(2)
# eigenvalue_analysis.plot_selected_eigenmode(3)
# eigenvalue_analysis.plot_selected_eigenmode(4)
# eigenvalue_analysis.plot_selected_eigenmode(5)
# eigenvalue_analysis.plot_selected_eigenmode(6)
# eigenvalue_analysis.plot_selected_eigenmode(7)

eigenvalue_analysis.plot_selected_first_n_eigenmodes(4)

eigenvalue_analysis.animate_selected_eigenmode(1)


# ===========================================
# Dynamic analysis 


'''
TODO: check kinematics at top point for various damping ratios
0.0, 0.001, 0.005, 0.01, 0.0125, 0.025, 0.05

will work only with 24 elements
'''

'''
NOTE: this analysis with the force_dynamic.npy can for now only be use
with 24 elements as loads are defined and recorded at 25 nodes
valid only for the pylon model
can be used for testing the caarc model as well

NOW AVAILABLE for
1 elements - 2 nodes
2 elements - 3 nodes
3 elements - 4 nodes
6 elements - 7 nodes
12 elements - 13 nodes
24 elements - 25 nodes
'''

possible_n_el_cases = [1, 2, 3, 6, 12, 24]
if beam_model.parameters['n_el'] not in possible_n_el_cases:
    err_msg = "The number of element input \"" + str(beam_model.parameters['n_el'])
    err_msg += "\" is not allowed for Dynamic Analysis \n"
    err_msg += "Choose one of: "
    err_msg += ', '.join([str(x) for x in possible_n_el_cases])
    raise Exception(err_msg)


# array_time = np.load(os.path.join('level_force','array_time.npy'))
# dynamic_force = np.load(os.path.join('level_force','force_dynamic' + str(beam_model.parameters['n_el']+1)+ '.npy'))
# dt = array_time[1] - array_time[0]

# # initial condition 
# # TODO all the inital displacement and velocity are zeros . to incorporate non zeros values required ? 
# dynamic_analysis = DynamicAnalysis(beam_model, dynamic_force, dt, array_time,
#                         "GenAlpha" )

# dynamic_analysis.solve()

# # reaction
# # forces
# # beam local Fy -> in CFD and OWC Fx
# dynamic_analysis.plot_reaction(1)
# # beam local Fz -> in CFD and OWC Fy
# dynamic_analysis.plot_reaction(2)
# # beam local Fx -> in CFD and OWC Fz
# dynamic_analysis.plot_reaction(0)
# # moments
# # beam local My -> in CFD and OWC Mx
# dynamic_analysis.plot_reaction(4)
# # beam local Mz -> in CFD and OWC My
# dynamic_analysis.plot_reaction(5)
# # beam local Mx -> in CFD and OWC Mz
# dynamic_analysis.plot_reaction(3)

# selected_time = 250
# dynamic_analysis.plot_selected_time(selected_time)

# dynamic_analysis.animate_time_history()

# # NOTE: for comparison the relevant DOFs have been selected
# # alongwind
# dynamic_analysis.plot_result_at_dof(-5, 'displacement')
# dynamic_analysis.plot_result_at_dof(-5, 'velocity')
# dynamic_analysis.plot_result_at_dof(-5, 'acceleration')

# # acrosswind
# dynamic_analysis.plot_result_at_dof(-4, 'displacement')
# dynamic_analysis.plot_result_at_dof(-4, 'velocity')
# dynamic_analysis.plot_result_at_dof(-4, 'acceleration')


# ============================================
# Static analysis 


# selected_time_step = 15000
# static_force = dynamic_force[:, selected_time_step]

# static_analysis = StaticAnalysis(beam_model)
# static_analysis.solve(static_force)
# static_analysis.plot_solve_result()