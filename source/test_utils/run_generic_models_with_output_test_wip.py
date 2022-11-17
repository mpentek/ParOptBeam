from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

# ==============================================
# Model choice

# NOTE: all currently available models

available_models = [
    'ProjectParameters3DGenericBuilding.json'
    ]

for available_model in available_models:

    # ==============================================
    # Parameter read
    with open(os_join(*['input', 'parameters', available_model]), 'r') as parameter_file:
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

# file check functionality

from source.auxiliary.compare_two_files_check import CompareTwoFilesCheck

file_checker_params ={
        "help"                  : "This process checks that two files are the same. This can be used in order to create tests, where a given solution is expected",
        "remove_output_file"    : True,
        "tolerance"             : 1e-6,
        "relative_tolerance"    : 1e-9
    }

file_name = 'dynamic_analysis_result_acceleration_for_dof_-1.dat'
file_checker_params["output_file_name"] = os_join(*['output','GenericBuilding',file_name])
file_checker_params["reference_file_name"] = os_join(*['output','GenericBuilding',file_name + '_ref'])

file_checker = CompareTwoFilesCheck(file_checker_params)
file_checker.execute()
