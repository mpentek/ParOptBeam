#%% IMPORTS
import json
from os.path import join as os_join

import numpy as np

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary
import source.ESWL.eswl_plotters as plotter_utilities
from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.ESWL.ESWL import ESWL
from source.model.structure_model import StraightBeam

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
    'ProjectParameters3D_CAARC_advanced.json',
    'ProjectParameters3DGenericBuildingUniform.json',
    'opt_ProjectParameters3DGenericBuildingUniform.json'
    ]

available_loads = [
    "input/force/generic_building/dynamic_force_4_nodes.npy",
    "input/force/generic_building/dynamic_90_force_4_nodes.npy", 
    "input/force/generic_building/dynamic_force_61_nodes.npy"
]

simple_uniform = [available_models[4]]
simple_uniform_opt = [available_models[5]]
symmetric_model = [available_models[0]]
unsymmetric_model = [available_models[2]]
CAARC_model = [available_models[3]]

dynamic_load_file = available_loads[0]
print ('...finished imports')
# ==============================================
#%% Model initialization - ESWL calculation - ESWL application - dynamic calculations
# READ AND CREATE THE BEAM MODEL
for available_model in simple_uniform_opt:
    
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
    print('\nNo need found for adapting structure for target values\n')


# ==============================================
# CREATE EIGENVALUE ANALYSIS
eigenvalue_analysis_in_parameters = False
for analyses_param in parameters['analyses_parameters']["runs"]:
    if analyses_param['type'] == "eigenvalue_analysis":
        eigenvalue_analysis = EigenvalueAnalysis(beam_model, analyses_param)
        eigenvalue_analysis.solve() #directly solve to have eigenforms etc available 
        eigenvalue_analysis_in_parameters = True

if not eigenvalue_analysis_in_parameters:
    raise Exception('no Eigenvalue Analysis in the Parameters')

# ==============================================
# # READ THE LOAD SIGNALS
load_signals_raw = np.load(get_adjusted_path_string(dynamic_load_file))
if len(load_signals_raw) != (beam_model.n_nodes*GD.DOFS_PER_NODE[beam_model.domain_size]):
    raise Exception('beam model and dynamic load signal have different number of nodes')
else:
    load_signals = auxiliary.parse_load_signal(load_signals_raw, GD.DOFS_PER_NODE[beam_model.domain_size], discard_time = 1000)

time_array = np.load('input\\force\\generic_building\\array_time.npy')
if time_array:
    dt = time_array[1] - time_array[0] # simulation time step
else: 
    dt = 0.1 
    
load_signals['sample_freq'] = 1/dt

# # plots of the load signals
#plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes, 1/1e+05)

# ==============================================
# CREATE ESWL FOR A RESPONSE 
response_labels = ['Qy', 'Qz', 'Mx', 'My', 'Mz']
load_directions = ['y','z','a','b','g']
resonant_load_directions = ['y','z','a'] #directions that have a mass that is moving in the eigenmode
include_all_rotations = True # this uses also for resonant calculations ['y','z','a','b','g']
lumped = False
# if false influences are computed using static analysis
plot_correlations = True
decoupled_influences = True
plot_influences = False

for response in response_labels[2:]:
    #response = 'My'
    # direction 'x' is left out for now since no response is related to it 

    eswl = ESWL(beam_model, eigenvalue_analysis, response, load_signals, resonant_load_directions, 
                load_directions, lumped, decoupled_influences, include_all_rotations)
    if plot_influences:
        plotter_utilities.plot_inluences(eswl)
    
    eswl.calculate_total_ESWL()

    if plot_correlations:
        plotter_utilities.plot_rho(eswl.BESWL.rho_collection_all, response)

    plot_eswl = True
    include_dynamic_results = True


    # ===============================================
    # RUN A STATIC ANALYSIS WITH THE ESWL 
    print('\nStatic analysis with ESWL...')
    eswl_vector = auxiliary.generate_static_load_vector_file(eswl.eswl_total[response])

    static_analysis = auxiliary.create_static_analysis_custom(beam_model, eswl_vector)
    static_analysis.solve()

    static_response = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[response]]

    # ==============================================
    # RUN A DYNAMIC ANALYSIS WITH ORIGINAL LOAD SIGNAL
    if include_dynamic_results:
        print('\nDynamic analysis with original dynamic load...')

        dynamic_analysis = auxiliary.create_dynamic_analysis_custom(beam_model, dynamic_load_file)
        dynamic_analysis.solve()

        response_id = GD.DOF_LABELS['3D'].index(GD.RESPONSE_DIRECTION_MAP[response])
        dynamic_response = dynamic_analysis.solver.dynamic_reaction[response_id]

        result_text = '\n'.join(('|'+response+'|' + ' with ESWL: '+ '{:.2e}'.format(abs(static_response[0]))+'\n',
                            response+ ' |max| dynamic: '+ '{:.2e}'.format(max(abs(dynamic_response)))+'\n',
                            '|ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                            'ESWL = '+ str( round( abs(static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn' ) )

        print ('\n','|'+response+'|', 'with ESWL: ', '{:.2e}'.format(abs(static_response[0])))
        print ('|',response, 'max| with dynamic: ', '{:.2e}'.format(max(abs(dynamic_response))))
        print ('|ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                'ESWL result = '+ str( round( abs(static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn result; should be > 1.00') 

        print ('\n... finished calculations')

    else:
        result_text = response + ' with ESWL: '+ '{:.2e}'.format(static_response[0])
        print (response, 'with ESWL: ', '{:.2e}'.format(static_response[0]))
    # ============================================================
    #%% Potsprocessing PLOTTING

    #plotter_utilities.plot_load_time_histories(load_signals, beam_model.nodal_coordinates)
    #plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes)

    available_components = ['mean', 'background', 'resonant', 'resonant_m','resonant_m_lumped','total', 'lrc', 'lrc1']
    # NOTE: all components use: ['all']
    if plot_eswl:
        #eswl.plot_eswl_directional_components(load_directions, response) # -> dictionary that has the loads for each reaction and each direction 
        #eswl.plot_eswl_directional_components(load_directions, response)['resonant', 'resonant_m']
        eswl.plot_eswl_components(response, load_directions, result_text, components_to_plot=['background', 'lrc', 'lrc1'])

    print ('... finished plotting')
    # %%
