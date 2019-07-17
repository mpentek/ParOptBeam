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
parameter_file = open('ProjectParameters3DPylonCadBeam.json', 'r')

''' 
Pylon model with geometry data from the sofistik beam
material parameters also from sofistik sheet 
'''
#parameter_file = open('ProjectParameters3DPylonSofiBeam.json', 'r')

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
#parameter_file = open('ProjectParameters3DCaarcBeamPrototype.json', 'r')


# ==============================================
# Parameter read


parameters = json.loads(parameter_file.read())
beam_model = StraightBeam(parameters)
# beam_model.plot_model_properties()


# ==============================================
# Eigenvalue analysis


'''
TODO: check eigenvalue analysis with number of elements
3, 6, 12, 24, 48, 96
'''

eigenvalue_analysis = EigenvalueAnalysis(beam_model)
eigenvalue_analysis.solve()
# eigenvalue_analysis.write_output_file()

# eigenvalue_analysis.plot_selected_eigenmode(1)
# eigenvalue_analysis.plot_selected_eigenmode(2)
# eigenvalue_analysis.plot_selected_eigenmode(3)
# eigenvalue_analysis.plot_selected_eigenmode(4)
# eigenvalue_analysis.plot_selected_eigenmode(5)
# eigenvalue_analysis.plot_selected_eigenmode(6)
# eigenvalue_analysis.plot_selected_eigenmode(7)

eigenvalue_analysis.plot_selected_first_n_eigenmodes(4)
# TODO: remedy animation bug
# eigenvalue_analysis.animate_selected_eigenmode(1)
# eigenvalue_analysis.animate_selected_eigenmode(3)


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
'''
array_time = np.load('array_time.npy')
dynamic_force = np.load('force_dynamic.npy')
dt = array_time[1] - array_time[0]

# initial condition 
# TODO all the inital displacement and velocity are zeros . to incorporate non zeros values required ? 
dynamic_analysis = DynamicAnalysis(beam_model, dynamic_force, dt, array_time,
                        "GenAlpha" )

dynamic_analysis.solve()

selected_time = 250
dynamic_analysis.plot_selected_time(selected_time)

# TODO: remedy animation bug
dynamic_analysis.animate_time_history()

# NOTE: for comparison the relevant DOFs have been selected
# alongwind
dynamic_analysis.plot_result_at_dof(145, 'displacement')
dynamic_analysis.plot_result_at_dof(145, 'velocity')
dynamic_analysis.plot_result_at_dof(145, 'acceleration')

# acrosswind
dynamic_analysis.plot_result_at_dof(146, 'displacement')
dynamic_analysis.plot_result_at_dof(146, 'velocity')
dynamic_analysis.plot_result_at_dof(146, 'acceleration')


# ============================================
# Static analysis 


selected_time_step = 15000
static_force = dynamic_force[:, selected_time_step]

static_analysis = StaticAnalysis(beam_model)
static_analysis.solve(static_force)
static_analysis.plot_solve_result()