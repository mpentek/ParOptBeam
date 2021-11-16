import json
from os.path import join as os_join
from os.path import sep as os_sep
import os

import numpy as np
import pickle as pkl

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary
import source.ESWL.eswl_plotters as plotter_utilities
from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.ESWL.ESWL import ESWL
from source.model.structure_model import StraightBeam

'''
1. get the load signals from the input.force
3. create a structure model and solve the eigenvalue problem
4. create the ESWL object; which itself creates B and R ESWL objects and calculates them
5. call a ESWL.solve() function the computes the loads and stores them in a dictionary with direction and specified response as keys
6. postprocess

TODO: nodal moments REWSL caclulated as modal inertial laods and by distributing the base moment along the height differ significantly
'''

available_models = {
    'symmetric_model':'ProjectParameters3DGenericBuilding.json',
    'unsymmetric_model':'ProjectParameters3DGenericBuilding_unsymmetric.json',
    'CAARC_model':'ProjectParameters3D_CAARC_advanced.json',
    'simple_uniform':'ProjectParameters3DGenericBuildingUniform.json',
    'simple_uniform_optimized4':'optimized' + os_sep +'opt_ProjectParameters3DGenericBuildingUniform_4_nodes_xi_1.0.json',
    'simple_uniform_optimized11_xi1.0':'optimized' + os_sep + 'opt_ProjectParameters3DGenericBuildingUniform_11_nodes_xi_1.0.json',
    'simple_uniform_optimized11_xi2.5':'optimized' + os_sep + 'opt_ProjectParameters3DGenericBuildingUniform_11_nodes_xi_2.5.json',
    'simple_uniform_optimized11_xi2.5_old':'opt_ProjectParameters3DGenericBuildingUniform_11_nodes_old.json',
    'simple_uniform_optimized_60':'opt_ProjectParameters3DGenericBuildingUniform_60.json'
}

available_loads = {
    '04_nodes_0_deg':os_join(*['input','force','generic_building','dynamic_force_4_nodes.npy']),
    '04_nodes_90_deg':os_join(*['input','force','generic_building','dynamic_90_force_4_nodes.npy']),
    '11_nodes_0_deg':os_join(*['input','force','generic_building','force_dynamic_0_turb_11.npy']),
    '16_nodes_0_deg':os_join(*['input','force','generic_building','force_dynamic_0_turb_16.npy']),
    '61_nodes_0deg':os_join(*['input','force','generic_building','dynamic_force_61_nodes.npy'])
}

discard_ramp_up = 2500 # first n steps 
time_array = np.load(os_join(*['input','force','generic_building','array_time.npy']))

# ==================================================================
# CURRENT SETTINGS ALL
# ==================================================================
damping_case = 1.0#2.5
if damping_case ==2.5:
    model_to_use = 'simple_uniform_optimized11_xi2.5'
    #model_to_use = 'simple_uniform_optimized11_xi2.5_old'
elif damping_case ==1.0:
    model_to_use = 'simple_uniform_optimized11_xi1.0'

model_to_use = 'simple_uniform'

dynamic_load_to_use = '11_nodes_0_deg'
dynamic_load_to_use = '04_nodes_0_deg'


load_directions_to_include = 'all'#['Fy','Fz','Mx']# for all other loops
load_directions_to_compute = 'automatic'#['Fy','Mz']#['Fz','My']# ['Mx']#,for the loop in calculate_eswl


response_labels_avail = ['Qy', 'Qz', 'Mx', 'My', 'Mz']
# for selected quantities list slices: Qy: :1, Qz: 1:2, Mx: 2:3, My: 3:4, Mz: 4:5
responses_to_analyse = response_labels_avail[4:5]#[3:4]#[2:3]#[1:2]#[:1]#
response_height = 0#120 # base reaction: 0 

optimize_gb = True
target = 'default'#'estimate'# # 'max_factor' #'default'
plot_objective_function = False
evaluate_gb = False

load_directions = ['y','z','a','b','g']
decoupled_influences = True # if false influences are computed using static analysis, this should directly incorporate coupling if it is present. Else just by simple mechanics
use_lrc = True # wheter lrc should be used for calculate total eswl if both are true lrc is used
use_gle = True # Kareems stuff
lrc_settings = {'lrc_method':True, 'lrc_corr_coeff':True, 'print_infos':False}
reswl_settings = {'type_to_combine':'modal_lumped', 'types_to_compute':{'base_moment_distr':True, 'modal_consistent':False, 'modal_lumped':True}}

plot_load_signals = False
plot_mode_shapes = False
plot_influences = False
plot_load_time_hist = False
plot_eswl = True
include_dynamic_results = True
available_components = ['mean', 'gle', 'resonant', 'resonant_m','resonant_m_lumped','total', 'lrc']
'''
NOTE: all components use: ['all']
    'gle': Kareems methods for background component
    'lrc': background component using LRC method
    'resonant': distribution of base moment e.g. Kareem eq. 29
    'resonant_m': modal inertial load consistent calculation Kareem eq. 27
    'resonant_m_lumped: modal inertial load using a nodal mass matrix/vector
'''
components_to_plot = ['resonant', 'lrc', 'mean','total']#]#, 'resonant_m_lumped',
#components_to_plot = ['lrc', 'gle','resonant_m_lumped','mean','total']#['gle', 'lrc']#, 
#components_to_plot = ['lrc', 'gle']#,'resonant_m_lumped','mean','total']#['gle', 'lrc']#, 

save_eswl_object = False
save_eswl_components = False
save_dyn_analysis = False
save_suffix = '' # if customly change the name a bit 
# ==============================================
# READ AND CREATE THE BEAM MODEL
# ==============================================
print ('---------- START ----------')
with open(os_join(*['input', 'parameters', available_models[model_to_use]]), 'r') as parameter_file:
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
dynamic_load_file = available_loads[dynamic_load_to_use]


# drop the first entries beloning to the ramp up
load_signals_raw = np.load(get_adjusted_path_string(dynamic_load_file))

if len(load_signals_raw) != (beam_model.n_nodes*GD.DOFS_PER_NODE[beam_model.domain_size]):
    raise Exception('beam model and dynamic load signal have different number of nodes')
else:
    # PARSING FOR ESWL 
    load_signals = auxiliary.parse_load_signal(load_signals_raw, GD.DOFS_PER_NODE[beam_model.domain_size], time_array,
                                                discard_time= discard_ramp_up, 
                                                load_directions_to_include = load_directions_to_include)
    # PARSING FOR DYNAMIC ANALYSIS
    dynamic_load_file = auxiliary.get_updated_dynamic_load_file(dynamic_load_file, discard_ramp_up, load_directions_to_include)

# # PLOTS OF LOAD SIGNALS
if plot_load_signals:
    plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes, discard_ramp_up)

# ==============================================
# RUN A DYNAMIC ANALYSIS
# ==============================================
# for comparison of results
print('\nDynamic analysis with dynamic load...')

dynamic_analysis = auxiliary.create_dynamic_analysis_custom(beam_model, dynamic_load_file)
dynamic_analysis.solve()
print ('...finished')


if save_dyn_analysis:
    # saving objects https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence/4529901 
    dest_path = ['source', 'ESWL', 'output']
    if damping_case ==1.0:
        dest_path.append('damping_001')
    filename = ['dyn',dynamic_load_to_use[:2],'nodes.pkl']
    if damping_case == 1.0:
        filename.insert(0, 'd1')

    filename = '_'.join(filename)
    dest_path.append(filename)
    with open(os_join(*dest_path),'wb') as dyn_output:
        pkl.dump(dynamic_analysis, dyn_output) #pkl.HIGHEST_PROTOCOL'damping_001',
    print ('\nsaved:', os_join(*dest_path))

# ===============================================
# CALCULATION OF ESWL FOR DIFFERENT RESPONSES
# ==============================================

for response in responses_to_analyse:
    # direction 'x' is left out for now since no response is related to it

    eswl = ESWL(beam_model, eigenvalue_analysis, response, response_height, load_signals, load_directions_to_include, load_directions_to_compute,
                decoupled_influences, plot_influences, use_lrc, use_gle, reswl_settings, 
                optimize_gb, target, evaluate_gb,
                plot_mode_shapes, dynamic_analysis.solver, plot_objective_function)

    if plot_influences:
        plotter_utilities.plot_influences(eswl)

    eswl.calculate_total_ESWL()

    # ===============================================
    # RUN A STATIC ANALYSIS WITH THE ESWL
    # ===============================================
    print('\nStatic analysis with ESWL...')
    eswl.evaluate_equivalent_static_loading()

    identifiers = [response,dynamic_load_to_use[:2],'nodes']
    opt_target_id = {'estimate':'gb_opt_est', 'quantile':'gb_opt_qnt', 'max_factor':'gb_opt_max' ,'default':'gb_default'}
    if optimize_gb:
        target_id = opt_target_id[target]
    else:
        target_id = 'gb_default'
        

    if use_lrc:
        identifiers.append('lrc')
    if use_gle:
        identifiers.append('gle')

    identifiers.append(target_id)
    if load_directions_to_include == 'all':
        identifiers.append('dirAll')
    else:
        identifiers.append('dirRed')

    if response_height != 0:
        identifiers.append('h'+str(response_height))
    
    identifiers[-1] += save_suffix + '.pkl'
    
    if save_eswl_object: 
        identifiers.insert(0, 'obj')
        if damping_case == 1.0:
            identifiers.insert(0, 'd1')
        object_filename = '_'.join(identifiers)
        dest_path = ['source', 'ESWL', 'output', object_filename]
        if damping_case == 1.0:
            dest_path.insert(-1, 'damping_001')
        with open(os_join(*['source', 'ESWL', 'output', object_filename]),'wb') as obj_output:
            pkl.dump(eswl, obj_output) #pkl.HIGHEST_PROTOCOL'damping_001',
        identifiers.pop(0)
        if damping_case == 1.0:
            identifiers.pop(0)
        print ('\nsaved:', os_join(*['source', 'ESWL', 'output', object_filename]))
    
    if save_eswl_components:
        identifiers.insert(0, 'comp')
        if damping_case == 1.0:
            identifiers.insert(0, 'd1')         
        comp_filename = '_'.join(identifiers)
        dest_path = ['source', 'ESWL', 'output', comp_filename]
        if damping_case == 1.0:
            dest_path.insert(-1, 'damping_001')

        with open(os_join(*dest_path),'wb') as comp_output:
            pkl.dump(eswl.eswl_components, comp_output) #pkl.HIGHEST_PROTOCOL'damping_001',
        print ('\nsaved:', os_join(*dest_path))

    # ==============================================
    # TAKE RESULT FROM DYNAMIC ANALYSIS AND PREPARE AN OUTPUT
    # ==============================================
    if include_dynamic_results:
        response_id = GD.DOF_LABELS['3D'].index(GD.RESPONSE_DIRECTION_MAP[response])
        dynamic_response = dynamic_analysis.solver.dynamic_reaction[response_id]

        result_text = '\n'.join(('|'+response+'|' + ' with ESWL: '+ '{:.2e}'.format(abs(eswl.static_response[0]))+'\n',
                            response+ ' |max| dynamic: '+ '{:.2e}'.format(max(abs(dynamic_response)))+'\n',
                            '|ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(eswl.static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                            'ESWL = '+ str( round( abs(eswl.static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn' ) )
        if optimize_gb:
            result_text += '; with optimized gb = ' + str([round(g_b, 2) for g_b in eswl.g_b_optimized])

        print ('\n    |'+response+'|', 'with ESWL: ', '{:.2e}'.format(abs(eswl.static_response[0])))
        print ('    |'+response+ ' max| with dynamic: ', '{:.2e}'.format(max(abs(dynamic_response))))
        print ('    |ESWL| - |Max_dyn|: '+ '{:.2e}'.format(abs(eswl.static_response[0])- max(abs(dynamic_response))) + ' should be positive' + '\n',
                '   ESWL result = '+ str( round( abs(eswl.static_response[0]) / max(abs(dynamic_response)),2 ) ) + ' of dyn result; should be > 1.00') 
        if optimize_gb:
            print ('    optimized g_b:', eswl.g_b_optimized)

        print ('\n... finished calculations for', response)

    else:
        result_text = response + ' with ESWL: '+ '{:.2e}'.format(eswl.static_response[0])
        print (result_text)
    # ============================================================
    # PLOTTING
    # ============================================================

    if plot_load_time_hist:
        plotter_utilities.plot_load_time_histories(load_signals, beam_model.nodal_coordinates)
        plotter_utilities.plot_load_time_histories_node_wise(load_signals, beam_model.n_nodes, discard_ramp_up)

    if plot_eswl:
        #eswl.plot_eswl_directional_load_components(eswl.eswl_total, eswl.structure_model.nodal_coordinates, load_directions, response) #
        eswl.plot_eswl_components(response, eswl.load_directions_to_compute, result_text, eswl.influences , components_to_plot=components_to_plot)

    print ('... finished plotting for', response)

print ('---------- END ----------')
