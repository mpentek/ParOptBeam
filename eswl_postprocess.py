
import numpy as np 
import matplotlib.pyplot as plt
from os.path import join as os_join
from os.path import sep
import pickle as pkl

import source.ESWL.eswl_plotters as eplt 
import source.ESWL.plot_settings as plot_settings
import source.ESWL.eswl_auxiliaries as auxiliary
import source.auxiliary.global_definitions as GD


''' 
should be used with objects or just dictionaries saved after the run_ESWL.py 
Things to evaluate:
    - total ESWL for different responses in one plot 
    - all/ some  components related to one reponse in one plot 
    - resonant components of a damped and less damped case
    - for a coupled case it would be interseting to see the total eswl of the different load components 
    - dynamic analysis in frequency domain

plot links:
ticks stuff: https://matplotlib.org/stable/gallery/ticks_and_spines/major_minor_demo.html 

''' 
src = {2.5:os_join(*['source','ESWL','output']), 1.0:os_join(*['source','ESWL','output','damping_001']), 'dyn':os_join(*['source','ESWL','output','dynamic_reactions'])}
file_prefix = {2.5:'',1.0:'d1_'}

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

width = eplt.cm2inch(7)
height = eplt.cm2inch(5)

mode = False
present = False

plot_params = plot_settings.get_params(width =width, height=height, use_tex = mode, minor_ticks=False, present = present)

plt.rcParams.update({'axes.formatter.limits':(-3,3)}) 
plt.rcParams.update(plot_params)

# # DEFINITIONS OF WHAT AND HOW TO PLOT
plot_options = {'show_plots':True,'savefig':mode,'savefig_latex':mode, 'update_plot_params':plot_params, 'present':present}
n_nodes_avail = ['4','11']
n_nodes = '11'

available_components = ['mean', 'gle', 'resonant', 'resonant_m','resonant_m_lumped','total', 'lrc']
gb_avail = {'estimate':'gb_opt_est','quantile':'gb_opt_qnt','default':'gb_default'}


damping_case = 2.5 # 2.5 or 1.0
#src = src[damping_case]

dyn_options = {'plot_dynamic_res':False, 'response_labels':['Mz','My'], 'damping_case':2.5, 'node_id':0, 'res_type':'reaction',
                'time_domain':True, 'freq_domain':False, 'include_both':True, 'log_plot':True, 'unit':'MN'}

eswl_total = {'plot_eswl':True, 'plot_load_distribution':True, 'rate_compare':True,
              'response_labels':['Mz','My','Mx'], 'load_directions':'automatic', 'gb':'default', 'unit':'KN', #,'Mx','My' load_directions: 'automatic
              'eswl_comp':['mean','lrc','resonant_m_lumped','total']}

compare_dyn_eswl = {'dyn_eswl':True, 'responses':['Mz','Qy','My','Qz','Mx'],'gb':'default', 'unit': 'MN','dyn_res':'dyn_qnt', 'dyn_res_2':'dyn_est', 'norm_by':'glob_max'} #'glob_max'#

eval_gb = {'eval_gb':False, 'gb_vs_m':False, 'eswl':True, 'distr':False, 'abs_gb':True,
            'gb_to_evaluate':['estimate','quantile'], 'resp':['Mz','Qy','My','Qz'], 'direction':['y','z'], 'unit': 'KN'}#,'Mx'['Mz','Qy','My','Qz']

compare_lrc_gle = {'lrc_gle':False, 'distribution':True,'quantify':False, 'h':['','_h120'], 'respone_label':['Qy','Qz'], 
                    'load_directions':'automatic', 'gb':'default', 'unit': 'KN'}

compare_resonant_options = {'compare_resonant':False, 'response_label':['Mz','My'], 'plot_fft':False, 'log_plot':False, 'plot_reswl':True, #
                            'unit':'KN', 'component':[available_components[4],available_components[-1]], 'load_direction':['y','z']}

plot_load_signals = {'plot':False, 'dynamic_load_file':'custom', 'direction':'y', 'add_mode':False}

# **********************************************************************************************************************************
# ******** ACTUAL LOADING AND PLOTTING OF RESULTS **********************************************************************************
# **********************************************************************************************************************************

if dyn_options['plot_dynamic_res']:
    src = src[dyn_options['damping_case']]
    with open(src + sep + file_prefix[dyn_options['damping_case']] + 'dyn_'+n_nodes +'_nodes.pkl', 'rb') as dynamic_analysis_input:
        dynamic_analysis = pkl.load(dynamic_analysis_input)

    if dyn_options['time_domain']:
        eplt.plot_dynamic_results(dynamic_analysis, dyn_options['response_labels'], dyn_options['node_id'], dyn_options['res_type'], add_fft=dyn_options['freq_domain'],
                                    include_extreme_value = True , include_fft=dyn_options['include_both'], log=dyn_options['log_plot'], unit=dyn_options['unit'],
                                    options=plot_options)
    if dyn_options['freq_domain']:
        eplt.plot_fft(dynamic_analysis= [dynamic_analysis], dof_label=dyn_options['response_labels'], log=dyn_options['log_plot'], options=plot_options)

if eswl_total['plot_eswl']:
    # fig size 14.8, 7.5
    src_i = src[2.5]
    eswl_resp = {}
    for resp_label in eswl_total['response_labels']:
        if eswl_total['load_directions'] == 'automatic':
            if plot_options['present']:
                load_directions = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED_RED[resp_label]
            else:
                load_directions = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[resp_label]
        else:
            load_directions = [eswl_total['load_directions']]
        # def plot_eswl_components(self, response_label, load_directions, textstr, influences , components_to_plot = ['all']):
        fname = src_i + sep + '_'.join(['obj', resp_label, n_nodes, 'nodes', 'lrc_gle', gb_avail[eswl_total['gb']],'dirAll.pkl'])
        with open(fname, 'rb') as eswl_input:
            eswl = pkl.load(eswl_input)

        eswl.evaluate_equivalent_static_loading()
        R_total = eswl.static_response[0]

        if eswl_total['plot_load_distribution']:

            eplt.plot_eswl_components(eswl.eswl_components,
                                        eswl.structure_model.nodal_coordinates,
                                        load_directions_to_include=load_directions,
                                        response_label = resp_label,
                                        textstr = None,
                                        influences = eswl.influences,
                                        components_to_plot = eswl_total['eswl_comp'],
                                        gb_label= gb_avail[eswl_total['gb']],
                                        go_final = True,
                                        R_total = R_total,#None,#
                                        unit=eswl_total['unit'],
                                        save_suffix= '_large',
                                        options=plot_options)

        eswl_resp[resp_label] = [eswl.eswl_components, R_total, load_directions, eswl.influences]
    
    if eswl_total['rate_compare']:
        # fig size: w7,h4 -> load_directions : 'automatic'
        eplt.plot_component_rate(eswl_resp, options = plot_options)

if compare_dyn_eswl['dyn_eswl']:
    #fig size: w7, h4.8
    src_i = src[2.5]
    
    result_dict = {'eswl':[], 'dyn_est':[], 'dyn_qnt':[], 'dyn_max':[], 'labels':[]}
    for response in compare_dyn_eswl['responses']:
        fname_dyn = src['dyn'] + sep + '_'.join(['kratos_dyn',response+'.npy'])
        
        fname_eswl = src_i + sep + '_'.join(['obj', response, n_nodes, 'nodes', 'lrc_gle', gb_avail[compare_dyn_eswl['gb']],'dirAll.pkl'])
        with open(fname_eswl, 'rb') as eswl_input:
            eswl = pkl.load(eswl_input)
        eswl.evaluate_equivalent_static_loading()
        dynamic_response = np.load(fname_dyn)
        result_dict['labels'].append(response)
        result_dict['eswl'].append(abs(eswl.static_response[0]))
        result_dict['dyn_est'].append(auxiliary.extreme_value_analysis_nist(dynamic_response, 0.02, response)[1])
        result_dict['dyn_qnt'].append(auxiliary.extreme_value_analysis_nist(dynamic_response, 0.02, response)[0])
        result_dict['dyn_max'].append(max(abs(dynamic_response)))

        if compare_dyn_eswl['norm_by']=='ref_vel':
            result_dict['norm_by'] = 'ref_vel'
            u = np.loadtxt(os_join(*['input','force','generic_building','ref180_build_H.dat']), usecols=2)[2500:]#ramp up 
            result_dict['u_mean'] = np.mean(u)
        elif compare_dyn_eswl['norm_by']=='glob_max':
            result_dict['norm_by'] = 'glob_max'
        
        to_compare = [compare_dyn_eswl['dyn_res'],'eswl']#,compare_dyn_eswl['dyn_res_2']]

    eplt.plot_eswl_dyn_compare(result_dict, compare=to_compare, options=plot_options)


if eval_gb['eval_gb']:
    ''' 
    results with different gb factors 
    ''' 
    if eval_gb['gb_vs_m']:
        # plotting gb versus M
        # fig size: w7, h4
        src_i = src[2.5] + sep + 'gb_eval' 
        res_arrs = []
        for i,resp in enumerate(eval_gb['resp']):
            fname = resp + '_' + eval_gb['direction'][i] + '.npy'
            res_arrs.append(np.load(src_i + sep + fname))
        eplt.plot_gb_eval(res_arrs, eval_gb['resp'], eval_gb['direction'], options=plot_options)

    if eval_gb['eswl']:
        src_i = src[2.5]
        result_dict = {}
        result_dict['gb_val'] = []
        result_dict['labels'] = []
        for i, resp_label in enumerate(eval_gb['resp']):
            result_dict[resp_label] = {}
            for gb in eval_gb['gb_to_evaluate']:
                
                fname = src_i + sep + '_'.join(['obj', resp_label, n_nodes, 'nodes', 'lrc_gle', gb_avail[gb],'dirAll.pkl'])
                with open(fname, 'rb') as eswl_input:
                    eswl_gb_i = pkl.load(eswl_input)

                result_dict[resp_label][gb] = eswl_gb_i.eswl_components[resp_label]
                nodal_coordinates = eswl_gb_i.structure_model.nodal_coordinates

                result_dict['labels'].append(resp_label)
                
                if eswl_gb_i.g_b_optimized[0]: 
                    result_dict['gb_val'].append(eswl_gb_i.g_b_optimized[0])
                else:
                    raise Exception('no gb optimized available - wrong opt type selected')

        if eval_gb['distr']:
            eplt.plot_gb_eval_eswl(result_dict, nodal_coordinates, unit=eval_gb['unit'], options=plot_options)
        if eval_gb['abs_gb']:
            # figsize w7 h4
            eplt.plot_gb_absolute(result_dict, options=plot_options)


if compare_lrc_gle['lrc_gle']:
    # figsize: 14.8, 8
    src_i = src[2.5]
    h_naming = {'':'base','_h120':'2/3H'}
    result_dict = {}
    infl = {}
    for resp_label in compare_lrc_gle['respone_label']:
        result_dict[resp_label] = {}
        infl[resp_label] = {}
        for h in compare_lrc_gle['h']:
            fname = src_i + sep + '_'.join(['obj', resp_label, n_nodes, 'nodes', 'lrc_gle', gb_avail[compare_lrc_gle['gb']],'dirAll'+h+'.pkl'])
            with open(fname, 'rb') as eswl_input:
                eswl_h_i = pkl.load(eswl_input)
            result_dict[resp_label][h_naming[h]] = eswl_h_i
            nodal_coordinates = eswl_h_i.structure_model.nodal_coordinates
            infl[resp_label][h_naming[h]] = eswl_h_i.influences[resp_label]

    if compare_lrc_gle['distribution']:
        eplt.plot_background_eval(result_dict, nodal_coordinates, directions=compare_lrc_gle['load_directions'],
                                    unit=compare_lrc_gle['unit'], options=plot_options)
    if compare_lrc_gle['quantify']:
        eplt.plot_quantify_background_methods(result_dict, nodal_coordinates, directions=compare_lrc_gle['load_directions'],
                                    unit=compare_lrc_gle['unit'], quantify = compare_lrc_gle['quantify'], influences=infl, options=plot_options)


if compare_resonant_options['compare_resonant']:
    
    if compare_resonant_options['plot_fft']:
        #figsize w14.8 h7 or 5
        # 2 nebeneinader: w7 h4.8
        resp_label = compare_resonant_options['response_label']#[0]
        with open(src[1.0] + sep + 'd1_dyn_'+n_nodes +'_nodes.pkl', 'rb') as dynamic_analysis_input_1:
            dynamic_analysis_1 = pkl.load(dynamic_analysis_input_1)
        with open(src[2.5] + sep +  'dyn_'+ n_nodes +'_nodes.pkl', 'rb') as dynamic_analysis_input_2:
            dynamic_analysis_2 = pkl.load(dynamic_analysis_input_2)
    
        eplt.plot_fft(dynamic_analysis= [dynamic_analysis_1, dynamic_analysis_2], damping_lables=[1.0,2.5], dof_label=resp_label,
                      log = compare_resonant_options['log_plot'], options=plot_options)
    
    if compare_resonant_options['plot_reswl']:
        # figsize w11 h7
        eswl_to_plot = {}
        for resp_label in compare_resonant_options['response_label']:
        
            with open(src[1.0] + sep + '_'.join(['d1_comp',resp_label, n_nodes ,'nodes_lrc_gle_gb_default_dirAll.pkl']), 'rb') as eswl_input_1:
                eswl_1 = pkl.load(eswl_input_1)
            with open(src[2.5] + sep +  '_'.join(['comp',resp_label, n_nodes ,'nodes_lrc_gle_gb_default_dirAll.pkl']), 'rb') as eswl_input_2:
                eswl_2 = pkl.load(eswl_input_2)
            eswl_to_plot[resp_label] = [eswl_1[resp_label], eswl_2[resp_label]]
        
        #eswl_list = [eswl_1[resp_label],eswl_2[resp_label]]
        eplt.plot_eswl_damping_compare(eswl_to_plot, [1.0,2.5], eswl_1['x_coords'], compare_resonant_options['component'], 
                                        compare_resonant_options['load_direction'], unit = compare_resonant_options['unit'],
                                        options=plot_options)

if plot_load_signals['plot']:
    ''' 
    nicely plot the signals along the height for the 4 nodes model
    '''
    dynamic_load_file = os_join(*['input','force','generic_building','dynamic_force_4_nodes.npy'])
    load_signals_raw = np.load(dynamic_load_file)
    time = np.load(os_join(*['input','force','generic_building','array_time.npy']))
    load_signals = auxiliary.parse_load_signal(load_signals_raw, dofs_per_node=6, time_array = time,
                                                discard_time= 1000)

    eplt.plot_load_time_histories_node_wise_figure(load_signals, n_nodes = 4, discard_time=0,  load_signal_labels=plot_load_signals['direction'], 
                                                    eigenmodes = plot_load_signals['add_mode'], options=plot_options)
    #eplt.plot_load_time_histories(load_signals, 4, 0, load_signal_labels=plot_load_signals['direction'], options=plot_options)

