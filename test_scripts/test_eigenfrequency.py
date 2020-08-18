from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

import numpy as np


# ==============================================
# run the different models


available_models = [
    'ProjectParameters3D_fixed-fixed_test.json',
    'ProjectParameters3D_fixed-free_test.json',
    'ProjectParameters3D_free-fixed_test.json',
    'ProjectParameters3D_fixed-pinned_test.json',
    'ProjectParameters3D_pinned-fixed_test.json',
    'ProjectParameters3D_pinned-pinned_test.json'
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

    # ==============================================
    # test the first 3 results against analytical solutions
    normed_tol = 1e-3

    name = parameters['model_parameters']['name']
    # result_path = "..\output" + str(name) +".txt"
    # result = np.loadtxt(res_path)
    # result = np.loadtxt(os_join(*['output',name,'eigenvalue_analysis_eigenmode_identification.dat']),
    # dtype={'names': ('counter','mode','type_counter','frequency','type'),'formats':('i','i','i','f','S12')}, usecols=(0,1,2,3,4))
    result = np.loadtxt(os_join(*['output',name,'eigenvalue_analysis_eigenmode_identification.dat']),
    dtype='i,f,S12', usecols=(2,3,4))  
    result_analytical = np.loadtxt(os_join(*['test_scripts','analytical_reference_results',name +'.txt']),
    dtype={'names': ('type_counter','frequency','type'),'formats':('i','f','S6')})
    for i_res_an in range(len(result_analytical)):
        type_counter = result_analytical[i_res_an][0]
        frequency = result_analytical[i_res_an][1]
        typ = result_analytical[i_res_an][2]
        for i_res in range(len(result)):
            if (result[i_res][0] == type_counter) & (result[i_res][2] == typ):
                try:
                    delta = abs(result[i_res][1]-frequency)/frequency
                    assert delta <= normed_tol
                    msg = "##################################################################################\n"
                    msg += name
                    msg += '\nTested Frequency:\n' + str(type_counter) + str(typ)
                    msg += '\npassed'
                    print(msg)

                except AssertionError:
                    msg = "##################################################################################\n"
                    msg += name
                    msg += '\nTested Frequency:\n' + str(type_counter) + str(typ)
                    msg += '\nFrequency is:\n' + str(result[i_res][1])
                    msg += '\nShould be:\n' + str(frequency)
                    msg += '\nDiffers by [%]:\n' + str(delta*100)
                    print(msg)
    


