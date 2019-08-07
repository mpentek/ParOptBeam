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
from source.analysis.analysis_controller import AnalysisController


# ==============================================
# Model choice

# NOTE: all currently available files
available_models = [
    # TODO: check model parameters for correctness
   'ProjectParameters3DPylonCadBeam.json',
    # with various elastic modulus
    'ProjectParameters3DPylonSofiBeam.json',
    'ProjectParameters3DPylonSofiBeamReducedE.json',
    # with elastic foundation
    'ProjectParameters3DPylonSofiBeamWithFoundationSoft.json',
    'ProjectParameters3DPylonSofiBeamWithFoundationMid.json',
    'ProjectParameters3DPylonSofiBeamWithFoundationHard.json',
    #
    'ProjectParameters3DCaarcBeam.json',
    #
    'ProjectParameters3DCaarcBeamPrototype.json',
    'ProjectParameters3DCaarcBeamPrototypeOptimizable.json']


for available_model in available_models:

    # ==============================================
    # Parameter read
    with open(os.path.join(*['input','parameters', available_model]), 'r') as parameter_file:
        parameters = json.loads(parameter_file.read())

    # create initial model
    beam_model = StraightBeam(parameters['model_parameters'])

    # additional changes due to optimization
    if 'optimization_parameters' in parameters:
        # return the model of the optimizable instance to preserve what is required by analyzis
        from source.model.optimizable_structure_model import OptimizableStraightBeam
        beam_model = OptimizableStraightBeam(
            beam_model, parameters['optimization_parameters']['adapt_for_target_values']).model
    else:
        print('No need found for adapting structure for target values')

    # ==============================================
    # Analysis wrapper

    analyses_controller = AnalysisController(
        beam_model, parameters['analyses_parameters'])
    analyses_controller.solve()
    analyses_controller.postprocess()
