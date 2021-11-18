# IMPORTS
import json
from os.path import join as os_join
import shutil
import re
import os

import numpy as np

import source.auxiliary.global_definitions as GD
from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
from source.model.structure_model import StraightBeam

# # 
# geht nicht da Ip von Iy + Iz berechnet wird und nicht as parameter gegebn wird im json
# dadurch ergibt sich immer erst mal eine falsche torsions frequenz. Ip muss dann optimiert werden
available_models = [
    'ProjectParameters3DGenericBuildingUniform_orig.json',
    'ProjectParameters3DGenericBuildingUniform.json',
    'optimized\\opt3_ProjectParameters3DGenericBuilding.json',
    'ProjectParameters3DGenericBuilding.json',
    'ProjectParameters3DGenericPylon.json',
    'ProjectParameters3DGenericBuilding_unsymmetric.json',
    'ProjectParameters3D_CAARC_advanced.json'
    ]


destination = os_join('..','input','parameters','optimized')

symmetric_model = [available_models[0]]
unsymmetric_model = [available_models[2]]
CAARC_model = [available_models[-1]]
symmetric_model = [available_models[1]]

# ==============================================
# Model initialization 
# READ AND CREATE THE BEAM MODEL
for available_model in symmetric_model:
    
    initial_model_file = os_join(*['input', 'parameters', available_model])
    #shutil.copyfile(initial_model, destination + 'opt_' + available_model)

    with open(initial_model_file, 'r') as parameter_file:
        parameters = json.loads(parameter_file.read())
    
    parameter_file.close()
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

    material = parameters["model_parameters"]['system_parameters']['material'] 
    intervals = parameters["model_parameters"]['system_parameters']['geometry']["defined_on_intervals"] #hier für jedes interval
    n_intervals = len(intervals)
    e_n = len(beam_model.elements)/n_intervals
    for i, e in enumerate(beam_model.elements):
        if i == 0:
            material['density'] = e.rho
            material['youngs_modulus'] = e.E
            intervals[i]["shear_area_y"] = [e.Asy]
            intervals[i]["shear_area_z"] = [e.Asz]
            intervals[i]["moment_of_inertia_y"] = [e.Iy]
            intervals[i]["moment_of_inertia_z"] = [e.Iz]
            intervals[i]["torsional_moment_of_inertia"] = [e.It]
            intervals[i]["opt_ip"] = [e.Ip]        
       
    parameters["model_parameters"]['system_parameters']['material'] = material
    parameters["model_parameters"]['system_parameters']['geometry']["defined_on_intervals"] = intervals

    # remove the optimization parameters 
    parameters.pop('optimization_parameters', None)

    s = re.split('[_.]', available_model)
    
    if os_join('optimized','opt') in re.split('[_.]', available_model):
        new_model = 'opt1_' + s[1] + '.json'
    else:
        new_model = 'opt_'+available_model

    new_file = open(os_join(*[destination, new_model]), 'w')
    json.dump(parameters, new_file)  
    new_file.close()
    print ('saved the optimized parameters in:', os_join(*[destination, new_model]))