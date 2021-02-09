import numpy as np
from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
from source.auxiliary.other_utilities import get_adjusted_path_string
import source.auxiliary.global_definitions as GD
from source.auxiliary.auxiliary_functionalities import parse_load_signal
from source.ESWL.ESWL import ESWL
import source.postprocess.plotter_utilities as plotter_utilities

# # somehow i need here the sturcture model, the load and response signal 
# # maybe also at this point i want to give the response i want to compute the loading for 

# 1. get the load signals from the input.force
# 2. get the response time history from a dynamic analysis or from other sources
# 3. create a structure model and solve the eigenvalue problem 
# 4. create the ESWL object; which itself creates B and R ESWL objects and calculates them 
# 5. call a ESWL.solve() function the computes the loads and stores them in a dictionary with direction and specified response as keys
# 6. postprocess

available_models = [
    'ProjectParameters3DGenericBuilding.json',
    'ProjectParameters3DGenericPylon.json',
    'ProjectParameters3DGenericBuilding_unsymmetric.json',
    'ProjectParameters3D_CAARC_advanced.json'
    ]
symmetric_model = [available_models[0]]
unsymmetric_model = [available_models[2]]
CAARC_model = [available_models[3]]

dynamic_load_file = "input/force/generic_building/dynamic_force_4_nodes.npy"

for available_model in CAARC_model:

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
    # Create an Eigenvalue Analysis
    eigenvalue_analysis_in_parameters = False
    for analyses in parameters['analyses_parameters']["runs"]:
        if analyses['type'] == "eigenvalue_analysis":
            eigenvalue_analysis = EigenvalueAnalysis(beam_model, analyses)
            eigenvalue_analysis.solve() #directly solve to have eigenforms etc available 
            eigenvalue_analysis_in_parameters = True

    if not eigenvalue_analysis_in_parameters:
        raise Exception('no Eigenvalue Analysis in the Parameters')

    # ==============================================
    # Read the load signals 
    load_signals_raw = np.load(get_adjusted_path_string(dynamic_load_file))
    if len(load_signals_raw) != (beam_model.n_nodes*GD.DOFS_PER_NODE[beam_model.domain_size]):
        raise Exception('beam model and dynamic load signal have different number of nodes')
    else:
        load_signals = parse_load_signal(load_signals_raw, GD.DOFS_PER_NODE[beam_model.domain_size], discard_time = 1000)
    
    # ==============================================
    # # plots of the load signals
    # plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes, 1/1e+05)

    # # Create the ESWL for a response 
    response = 'Mz'
    load_directions = ['y', 'z','a']

    eswl = ESWL(beam_model, eigenvalue_analysis, response, load_signals, load_directions)
    eswl.calculate_total_ESWL()
    
    eswl.plot_eswl_load_components(['y','z','a'], response) # -> dictionary that has the loads for each reaction and each direction 

