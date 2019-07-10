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

parameter_file = open('ProjectParameters3DBeam.json', 'r')
parameters = json.loads(parameter_file.read())

beam_model = StraightBeam(parameters)

# ==============================================
# #eigen value analysis
eigenvalue_analysis = EigenvalueAnalysis(beam_model)
eigenvalue_analysis.solve()
# eigenvalue_analysis.write_output_file()
# eigenvalue_analysis.plot_selected_eigenmode(1)
# eigenvalue_analysis.plot_selected_eigenmode(4)
# eigenvalue_analysis.plot_selected_first_n_eigenmodes(3)
# eigenvalue_analysis.animate_selected_eigenmode(1)


# ============================================
# #static analysis 
#static_force = 100000 * np.ones(150)
static_force = np.zeros(150)
static_force[127] = 10000
static_analysis = StaticAnalysis(beam_model)
static_analysis.solve(static_force)
static_analysis.plot_solve_result()
static_analysis.write_output_file()
# ===========================================
# Dynamic analysis 
# array_time = np.load('array_time.npy')
# dynamic_force = np.load('force_dynamic.npy')
# dt = array_time[1] - array_time[0]
# # initial condition # TODO all the inital displacement and velocity are zeros . to incorporate non zeros values required ? 
# dynamic_analysis = DynamicAnalysis(beam_model, dynamic_force, dt, array_time,
#                         "GenAlpha" )
# dynamic_analysis.solve()
# dynamic_analysis.plot_selected_time_step(125)
# dynamic_analysis.animate_time_history()
# dynamic_analysis.write_result_at_dof(31, 'displacement')
# dynamic_analysis.plot_result_at_dof(31, 'acceleration')
# dynamic_analysis.plot_result_at_dof(31, 'displacement')