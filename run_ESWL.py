import json
from os.path import join as os_join
import os

import numpy as np

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary
import source.ESWL.eswl_plotters as plotter_utilities
from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.ESWL.ESWL import ESWL
from source.model.structure_model import StraightBeam

# 1. get the load signals from the input.force
# 3. create a structure model and solve the eigenvalue problem 
# 4. create the ESWL object; which itself creates B and R ESWL objects and calculates them 
# 5. call a ESWL.solve() function the computes the loads and stores them in a dictionary with direction and specified response as keys
# 6. postprocess

available_models = [
    'ProjectParameters3DGenericBuilding.json',
    'ProjectParameters3DGenericBuilding_unsymmetric.json',
    'ProjectParameters3D_CAARC_advanced.json',
    'ProjectParameters3DGenericBuildingUniform.json',
    'opt_ProjectParameters3DGenericBuildingUniform.json',
    'opt_ProjectParameters3DGenericBuildingUniform_60.json'
    ]

available_loads = [
    "input\\force\\generic_building\\dynamic_force_4_nodes.npy",
    "input\\force\\generic_building\\dynamic_90_force_4_nodes.npy", 
    "input\\force\\generic_building\\dynamic_force_61_nodes.npy"
]

discard_ramp_up = 2000
time_array = np.load('input\\force\\generic_building\\array_time.npy')

symmetric_model = [available_models[0]]
unsymmetric_model = [available_models[1]]

CAARC_model = [available_models[2]]

# # one interval only 
simple_uniform = [available_models[3]]
# with optimized 
simple_uniform_optimized = [available_models[4]]

# # for a load file with 61 nodes
simple_uniform_optimized_60 = [available_models[5]]

dynamic_load_file = available_loads[0]

<<<<<<< HEAD

=======
>>>>>>> feature_torsion_coupling
plot_load_signals = False

# ==============================================
# READ AND CREATE THE BEAM MODEL
# ==============================================
for available_model in simple_uniform_optimized:
    
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
# ==============================================
eigenvalue_analysis_in_parameters = False
for analyses_param in parameters['analyses_parameters']["runs"]:
    if analyses_param['type'] == "eigenvalue_analysis":
        eigenvalue_analysis = EigenvalueAnalysis(beam_model, analyses_param)
        eigenvalue_analysis.solve() #directly solve to have eigenforms etc available 
        eigenvalue_analysis_in_parameters = True

if not eigenvalue_analysis_in_parameters:
    raise Exception('no Eigenvalue Analysis in the Parameters')


# ==============================================
# READ THE LOAD SIGNALS
# ==============================================

# drop the first entries beloning to the ramp up
load_signals_raw = np.load(get_adjusted_path_string(dynamic_load_file))[:,discard_ramp_up:]
<<<<<<< HEAD

dynamic_load_file_ramp_up = auxiliary.discard_ramp_up(dynamic_load_file)
np.save(dynamic_load_file_ramp_up, load_signals_raw)
=======
>>>>>>> feature_torsion_coupling
if len(load_signals_raw) != (beam_model.n_nodes*GD.DOFS_PER_NODE[beam_model.domain_size]):
    raise Exception('beam model and dynamic load signal have different number of nodes')
else:
    # create a dictionary with directions as keys
    load_signals = auxiliary.parse_load_signal(load_signals_raw, GD.DOFS_PER_NODE[beam_model.domain_size], time_array)

# # PLOTS OF LOAD SIGNALS
if plot_load_signals:
    plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes, discard_ramp_up)
<<<<<<< HEAD

# ==============================================
# RUN A DYNAMIC ANALYSIS 
# ==============================================
# for comparison of results
print('\nDynamic analysis with original dynamic load')

if discard_ramp_up:
    dynamic_load_file = dynamic_load_file_ramp_up

dynamic_analysis = auxiliary.create_dynamic_analysis_custom(beam_model, dynamic_load_file)
dynamic_analysis.solve()
=======

# ==============================================
# RUN A DYNAMIC ANALYSIS 
# ==============================================
# for comparison of results
print('\nDynamic analysis with original dynamic load')

# TODO: no discarded ramp up here, guess that it is not influencing the result signigicantly 
dynamic_analysis = auxiliary.create_dynamic_analysis_custom(beam_model, dynamic_load_file)
dynamic_analysis.solve()

>>>>>>> feature_torsion_coupling


# ==============================================
# INPUT SETTINGS FOR THE ESWL OBJECT
# ==============================================
<<<<<<< HEAD
response_labels_avail = ['Qy', 'Qz', 'Mx', 'My', 'Mz']
# for selected quantities list slices: Qy: :1, Qz: 1:2, Mx: 2:3, My: 3:4, Mz: 4:5
responses_to_analyse = response_labels_avail[4:5]
=======
# INPUT SETTINGS FOR THE ESWL OBJECT
# ==============================================
response_labels_avail = ['Qy', 'Qz', 'Mx', 'My', 'Mz']
# for selected quantities list slices: Qy: :1, Qz: 1:2, Mx: 2:3, My: 3:4, Mz: 4:5
responses_to_analyse = response_labels_avail[3:5]
>>>>>>> feature_torsion_coupling

load_directions = ['y','z','a','b','g']
decoupled_influences = False # if false influences are computed using static analysis, this should directly incorporate coupling if it is present. Else just by simple mechanics
use_lumped = False
use_lrc = False # wheter lrc should be used for calculate total eswl

plot_mode_shapes = False
plot_correlations = False
plot_influences = False
plot_load_time_hist = False
plot_eswl = True
include_dynamic_results = True
available_components = ['mean', 'background', 'resonant', 'resonant_m','resonant_m_lumped','total', 'lrc']
'''
NOTE: all components use: ['all']
    'background': Kareems methods
    'lrc': background component using LRC method
    'resonant': distribution of base moment e.g. Kareem eq. 29
    'resonant_m': modal inertial load consisten calculation Kareem eq. 27
    'resonant_m_lumped: modal inertial using a nodal mass matrix/vector
''' 
<<<<<<< HEAD
components_to_plot = ['background', 'lrc', 'mean']
=======
components_to_plot = ['resonant', 'resonant_m','background', 'mean']
>>>>>>> feature_torsion_coupling

# ===============================================
# CALCULATION OF ESWL FOR DIFFERENT RESPONSES
# ==============================================

for response in responses_to_analyse:
    # direction 'x' is left out for now since no response is related to it 

    eswl = ESWL(beam_model, eigenvalue_analysis, response, load_signals,load_directions, 
                use_lumped, decoupled_influences, use_lrc, plot_mode_shapes)

    if plot_influences:
        plotter_utilities.plot_inluences(eswl)
    
    eswl.calculate_total_ESWL()

    if plot_correlations:
        plotter_utilities.plot_rho(eswl.BESWL.rho_collection_all, response)

    # ===============================================
    # RUN A STATIC ANALYSIS WITH THE ESWL 
    print('\nStatic analysis with ESWL...')
    eswl_vector = auxiliary.generate_static_load_vector_file(eswl.eswl_total[response])

    static_analysis = auxiliary.create_static_analysis_custom(beam_model, eswl_vector)
    static_analysis.solve()

    static_response = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[response]]

    # ==============================================
    # TAKE RESULT FROM DYNAMIC ANALYSIS AND PREPARE AN OUTPUT
    if include_dynamic_results:
        response_id = GD.DOF_LABELS['3D'].index(GD.RESPONSE_DIRECTION_MAP[response])
        dynamic_response = dynamic_analysis.solver.dynamic_reaction[response_id]

        result_text = '\n'.join(('|'+response+'|' + ' with ESWL: '+ '{:.2e}'.format(abs(static_response[0]))+'\n',
                            response+ ' |max| dynamic: '+ '{:.2e}'.format(max(abs(dynamic_response)))+'\n',
                            '|ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                            'ESWL = '+ str( round( abs(static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn' ) )

        print ('\n','|'+response+'|', 'with ESWL: ', '{:.2e}'.format(abs(static_response[0])))
        print ('|'+response+ ' max| with dynamic: ', '{:.2e}'.format(max(abs(dynamic_response))))
        print ('|ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                'ESWL result = '+ str( round( abs(static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn result; should be > 1.00') 

        print ('\n... finished calculations')

    else:
        result_text = response + ' with ESWL: '+ '{:.2e}'.format(static_response[0])
        print (result_text)
    # ============================================================
    # PLOTTING
    # ============================================================

    if plot_load_time_hist:
        plotter_utilities.plot_load_time_histories(load_signals, beam_model.nodal_coordinates)
        plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes)

    if plot_eswl:
        #eswl.plot_eswl_directional_load_components(eswl.eswl_total, eswl.structure_model.nodal_coordinates, load_directions, response) # 
        eswl.plot_eswl_components(response, load_directions, result_text, eswl.influences ,components_to_plot=components_to_plot)

    print ('... finished plotting')
    