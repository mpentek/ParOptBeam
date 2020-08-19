from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

import numpy as np
# ==============================================
# create input file
parameters = {
    "model_parameters": {
        "name": "name",
        "domain_size": "3D",
        "system_parameters": {
            "element_params": {
                "type": "Bernoulli",
                "is_nonlinear": False
            },
            "material": {
                "is_nonlinear": False,
                "density": 7850.0,
                "youngs_modulus": 2.10e11,
                "poisson_ratio": 0.3,
                "damping_ratio": 0.0
            },
            "geometry": {
                "length_x": 12,
                "number_of_elements": 10,
                "defined_on_intervals": [{
                    "interval_bounds" : [0.0,"End"],
                    "length_y": [0.20],
                    "length_z": [0.40],
                    "area"    : [0.08],
                    "shear_area_y" : [0.0667],
                    "shear_area_z" : [0.0667],
                    "moment_of_inertia_y" : [0.0010667],
                    "moment_of_inertia_z" : [0.0002667],
                    "torsional_moment_of_inertia" : [0.00007328]}]
            }
        },
        "boundary_conditions": "bc"
    },
    "analyses_parameters":{
        "global_output_folder" : "some/path",
        "model_properties": {
            "write": False,
            "plot": False
        },
        "report_options": {
            "combine_plots_into_pdf" : False,
            "display_plots_on_screen" : False,
            "use_skin_model" : False
        },
        "runs": [{
                "type": "eigenvalue_analysis",
                "settings": {
                    "normalization": "mass_normalized"},
                "input":{},
                "output":{
                    "eigenmode_summary": {
                        "write" : False, 
                        "plot" : False},
                    "eigenmode_identification": {
                        "write" : True, 
                        "plot" : False},
                    "selected_eigenmode": {
                        "plot_mode": [], 
                        "write_mode": [],
                        "animate_mode": [],
                        "animate_skin_model": []},
                    "selected_eigenmode_range": {
                        "help": "maximum 4 modes per range at a time",
                        "considered_ranges": [[1,2]], 
                        "plot_range": [False, False], 
                        "write_range": [False, False]}
                    }
            
            }]
    }
}


# ==============================================
# run and test the different models


available_models = [['FixedFixedTest','fixed-fixed'],
                    ['FreeFixedTest','free-fixed'],
                    ['FixedFreeTest','fixed-free'],
                    ['FixedPinnedTest','fixed-pinned'],
                    ['PinnedFixedTest','pinned-fixed'],
                    ['PinnedPinnedTest','pinned-pinned']]

for available_model in available_models:

    # ==============================================
    # Set BC and name
    parameters['model_parameters']['name']=available_model[0]
    parameters['model_parameters']['boundary_conditions']=available_model[1]

    # create initial model
    beam_model = StraightBeam(parameters['model_parameters'])

    # ==============================================
    # Analysis wrapper

    analyses_controller = AnalysisController(
        beam_model, parameters['analyses_parameters'])
    analyses_controller.solve()
    analyses_controller.postprocess()

    # ==============================================
    # test results against available analytical solutions
    abs_tol = 1e-4

    name = parameters['model_parameters']['name']
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
                    delta = abs(result[i_res][1]-frequency)
                    assert delta <= abs_tol
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
                    msg += '\nDiffers by:\n' + str(delta)
                    print(msg)
    


