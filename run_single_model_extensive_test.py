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
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_wrapper import AnalysisWrapper


# ==============================================
# Model choice

# NOTE: using this single (yet extensive) file for testing
available_models = ['ProjectParameters3DCaarcBeamPrototypeWithOuttriger.json']

for available_model in available_models:

    # ==============================================
    # Parameter read
    with open(os.path.join(*['input','parameters', available_model]), 'r') as parameter_file:
        parameters = json.loads(parameter_file.read())

    # create initial model
    beam_model = StraightBeam(parameters['model_parameters'])

    # plot initial model properties
    # beam_model.plot_model_properties()

    # additional changes due to optimization
    if 'optimization_parameters' in parameters:
        # return the model of the optimizable instance to preserve what is required by analyzis
        from source.model.optimizable_structure_model import OptimizableStraightBeam
        beam_model = OptimizableStraightBeam(
            beam_model, parameters['optimization_parameters']['adapt_for_target_values']).model
    else:
        print('No need found for adapting structure for target values')

    # plot optimized model properties
    # beam_model.plot_model_properties()

    # ==============================================
    # Analysis wrapper

    analyses_controller = AnalysisWrapper(
        beam_model, parameters['analyses_parameters'])
    analyses_controller.solve()
    analyses_controller.postprocess()

    # # ==============================================
    # # Eigenvalue analysis
    # '''
    # TODO: check eigenvalue analysis with number of elements
    # 3, 6, 12, 24, 48, 96
    # '''

    # # ===========================================
    # # Dynamic analysis

    # '''
    # TODO: check kinematics at top point for various damping ratios
    # 0.0, 0.001, 0.005, 0.01, 0.0125, 0.025, 0.05
    # '''

    # '''
    # NOTE: works on with 1, 2, 3, 6, 12, 24 elements
    # valid only for the pylon model
    # can be used for testing the caarc model as well

    # # ============================================
    # # Static analysis
