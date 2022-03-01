from os.path import join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController


# ==============================================
# Model choice

# NOTE: all currently available files
parametric_runs = {
    'input/force/caarc/0_turb/force_dynamic_0_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont0.json',
             'ProjectParameters3DCaarcBeamInt0.json',
             'ProjectParameters3DCaarcBeamIntOut0.json']},
    'input/force/caarc/45_turb/force_dynamic_45_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont45.json',
             'ProjectParameters3DCaarcBeamInt45.json',
             'ProjectParameters3DCaarcBeamIntOut45.json']},
    'input/force/caarc/90_turb/force_dynamic_90_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']},
    'input/force/caarc/90_no_turb/force_dynamic_90_no_turb':
        {'output_folder_prefix': 'no_turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']}}

# TODO: add 45 degree setup

# ==============================================
# Parametric run

for damping_ratio in ['0.000', '0.025']:
    for load_file, parametric_run in parametric_runs.items():

        for available_model in parametric_run['project_params']:

            # ==============================================
            # Parameter read
            with open(join(*['input', 'parameters', 'caarc', available_model]), 'r') as parameter_file:
                parameters = json.loads(parameter_file.read())

            for n_el in [60, 30, 15, 3, 2, 1]:
                parameters["model_parameters"]["system_parameters"]["geometry"]["number_of_elements"] = n_el
                parameters["model_parameters"]["system_parameters"]["material"]["damping_ratio"] = float(
                    damping_ratio)

                global_output_folder = join(*['Caarc',
                                              parametric_run['output_folder_prefix'],
                                              damping_ratio.replace('.', '_'),
                                              parameters['model_parameters']['name'],
                                              str(n_el)])

                parameters["analyses_parameters"]["global_output_folder"] = global_output_folder

                for idx, parameter in enumerate(parameters["analyses_parameters"]["runs"]):
                    if parameter["type"] == "dynamic_analysis":
                        load_file_path = load_file + str(n_el + 1) + '.npy'
                        load_file_path = join(*load_file_path.split('/'))
                        parameters["analyses_parameters"]["runs"][idx]["input"]["file_path"] = load_file_path

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
