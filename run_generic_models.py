from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController


# ==============================================
# Model choice

# NOTE: all currently available models

available_models = [
    'ProjectParameters3DGenericBuilding.json',
    'ProjectParameters3DGenericPylon.json',
    'ProjectParameters3DGenericBuilding_unsymmetric.json',
    'ProjectParameters3D_CAARC_advanced.json'
    ]
symmetric_model = [available_models[0]]
unsymmetric_model = [available_models[2]]
CAARC_model = [available_models[3]]

for available_model in symmetric_model:

    # ==============================================
    # Parameter read
    with open(os_join(*['input', 'parameters', available_model]), 'r') as parameter_file:
        parameters = json.loads(parameter_file.read())

    # for changing some parameters to see what generally happens
    for parameter in parameters['model_parameters']['system_parameters']['geometry']['defined_on_intervals']:

        parameter['eccentricity_z'][0] *= 1
        parameter['torsional_moment_of_inertia'][0] *= 1
       
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

    analyses_controller = AnalysisController(beam_model, parameters['analyses_parameters'])
    analyses_controller.solve()
    analyses_controller.postprocess()