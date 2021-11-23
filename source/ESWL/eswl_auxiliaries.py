import numpy as np
import os
from os.path import sep as os_sep
from os.path import join as os_join
import pickle as pkl

import source.auxiliary.global_definitions as GD
import source.auxiliary.statistics_utilities as stats_utils

# # ========= ESWL =============
def rms(signal, subtract_mean = True):
    '''
    returns the root mean square of a signal
    '''
    if subtract_mean:
        result = np.sqrt(np.mean(np.square(signal - np.mean(signal))))
    else:
        result = np.sqrt(np.mean(np.square(signal)))
    return result

def cov_custom(signal1, signal2):

    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    cov = 0.0
    for i in range(len(signal1)):
        cov += (signal1[i] - mean1)*(signal2[i] - mean2)

    return cov/len(signal1)

def parse_load_signal(signal_raw, dofs_per_node, time_array = None, load_directions_to_include = 'all', discard_time = None):
    '''
    - sorts the load signals in a dictionary with load direction as keys:
        x,y,z: nodal force
        a,b,g: nodal moments
    - deletes first entries until discard_time
    - only gives back the components specified, default is all 
    TODO: instead of setting unused direction to 0 do some small number or exclude them differntly 
    '''
    if discard_time:
        signal_raw = signal_raw[:,discard_time:]
    if load_directions_to_include == 'all':
        load_directions_to_include = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    n_nodes = int(signal_raw.shape[0]/GD.DOFS_PER_NODE['3D'])
    if dofs_per_node != 6:
        raise Exception('load signal parsing only for 6 dofs per node - check dynamic load files')
    else:
        signal = {}
        for i, label in enumerate(GD.DOF_LABELS['3D']):
            if GD.DIRECTION_LOAD_MAP[label] in load_directions_to_include:
                signal[label] = signal_raw[i::dofs_per_node]
            else:
                signal[label] = np.zeros((n_nodes,signal_raw.shape[1]))

    if time_array:
        dt = time_array[1] - time_array[0] # simulation time step
    else:
        dt = 0.1 # some default

    signal['sample_freq'] = 1/dt

    return signal

def parse_load_signal_backwards(signal):
    '''
    signal kommt als dictionary mit den Richtungen als keys (müssen nicht alle 6 Richtugne sein)
    output soll row vector sein mit dofs * n_nodes einträgen
    '''
    shape = GD.DOFS_PER_NODE['3D'] * len(list(signal.values())[0])
    signal_raw = np.zeros(shape)
    for label in signal:
        dof_label_id = GD.DOF_LABELS['3D'].index(label)
        for i, val in enumerate(signal[label]):
            sort_id = i * GD.DOFS_PER_NODE['3D'] + dof_label_id
            signal_raw[sort_id] = val

    #TODO: must this be transposed?!
    return signal_raw

def rename_load_file(load_file, load_directions_to_include):
    '''
    return a modified file name with path to save
    '''
    if load_directions_to_include == 'all':
        d = '_all'
    else:
        d = '_red'
    new_name = load_file.split(os.path.sep)[-1].split('.')[0] + '_ramp_up' + d + '.npy'
    dest = os.path.join(*load_file.split(os.path.sep)[:-1])
    return os.path.join(dest, new_name)

def get_updated_dynamic_load_file(dynamic_load_file, discard_time, load_directions_to_include = 'all'):
    '''
    updates the load file according to 
        ramp up time 
        load directions to include
    returns the file name /path to it which is used by a dynamic analysis. Suffixes are added if changes are made:
        - _ramp_up: if a ramp up is discarded
        - _all/_red: if a reduced number (red) of load directions is used or all are kept (all)
    ''' 
    ramp_up, directions = False, False
    signal_raw = np.load(dynamic_load_file)
    if discard_time:
        signal_raw = signal_raw[:,discard_time:]
        ramp_up = True
    if load_directions_to_include != 'all':
        new_signal = np.zeros(signal_raw.shape)
        for load in load_directions_to_include:
            load_id = GD.DOF_LABELS['3D'].index(GD.LOAD_DIRECTION_MAP[load])
            new_signal[load_id::GD.DOFS_PER_NODE['3D']] = signal_raw[load_id::GD.DOFS_PER_NODE['3D']]
        directions = True
        signal_raw = new_signal

    if not ramp_up and not directions:
        return dynamic_load_file
    else:
        dynamic_load_file_ramp_up = rename_load_file(dynamic_load_file, load_directions_to_include)
        np.save(dynamic_load_file_ramp_up, signal_raw)
        return dynamic_load_file_ramp_up

def generate_static_load_vector_file(load_vector):
    '''
    return a npy file and saves it for a given load.

    If load is given as dictionary with directions as keys it parses it to a 1D array that is needed in the ParOptBeam
    '''
    if isinstance(load_vector, dict):
        load = parse_load_signal_backwards(load_vector)
    else:
        load = load_vector

    src_path = os_join('input','force','generic_building','eswl')
    if not os.path.isdir(src_path):
        os.mkdir(src_path)

    file_name = src_path + os_sep + 'eswl_' + str(int(len(load)/GD.DOFS_PER_NODE['3D'])) + '_nodes.npy'

    np.save(file_name, load)

    return file_name

def reduce_nodes_of_dynamic_load_file(file_name):
    f_name_60 = 'force_dynamic_90_turb_61.npy'

    force_60 = np.load(f_name_60)

    z_4 = np.linspace(0,180,4)
    z_60 = np.linspace(0,180,61)

    force_red = np.zeros((24,force_60.shape[1]))
    dofs = ['x', 'y', 'z', 'a', 'b', 'g']

    for dof, label in enumerate(dofs):
        dof_loads = force_60[dof::6]
        step =  int((z_60.shape[0] - 1) / (z_4.shape[0]-1))
        for elidx in range(z_4.shape[0]- 1):
            start = int(step * elidx)
            mid = int(start + 0.5*step)
            end = int(start + step)
            # sum and alpha

            node0 = dof_loads[start:mid]
            node0_sum = sum(node0)

            force_red[dof+elidx*6] = sum(dof_loads[start:mid])
            force_red[dof+(elidx+1)*6] = sum(dof_loads[mid:end])

            if label in ['b','g']:
                #z4_0 = np.asarray([z_4[elidx]]*int(step/2))
                dx0 = np.transpose(z_60[:int(0.5*step)])
                dx1 = -dx0

                f0 = dof_loads[start:mid]
                f1 = dof_loads[mid:end]

                for i in range(len(dx0)):
                    f0[i] *= dx0[i]
                    f1[i] *= dx1[i]

                force_red[dof+elidx*6] += sum(f0)
                force_red[dof+(elidx+1)*6] += sum(f1)

    np.save('90_dynamic_force_4_nodes.npy', force_red)

def generate_unit_nodal_force_file(number_of_nodes, node_of_load_application, force_direction, magnitude):
    '''
    creating a force .npy file with a nodal force at given node, direction and magnitude
    '''
    src_path = os.path.join(*['input','force','generic_building','unit_loads'])
    domain_size = '3D'

    loaded_dof = (node_of_load_application)*GD.DOFS_PER_NODE[domain_size] + GD.DOF_LABELS[domain_size].index(force_direction)

    force_data = np.zeros(GD.DOFS_PER_NODE[domain_size]*number_of_nodes)
    force_data[loaded_dof] += magnitude
    force_data = force_data.reshape(GD.DOFS_PER_NODE[domain_size]*number_of_nodes,1)

    force_file_name = src_path + os.path.sep + 'unit_static_force_' + str(number_of_nodes) + '_nodes_at_' + str(node_of_load_application) + \
                        '_in_' + force_direction+'.npy'
    np.save(force_file_name, force_data)

    return force_file_name

def get_influence(structure_model, load_direction, node_id, response, response_node_id = 0):
    '''
    influence function representing the response R due to a unit load acting at elevation z along load direction s
    computed using the beam model and a static analysis with load vector of zeros just 1 at node_id
    NOTE: Sofar returning always reaction at ground node
    '''
    src_path = os.path.join(*['input','force','generic_building','unit_loads'])

    needed_force_file = src_path + os.path.sep + 'unit_static_force_' + str(structure_model.n_nodes) + \
                        '_nodes_at_' + str(node_id) + \
                        '_in_' + load_direction+'.npy'

    if os.path.isfile(needed_force_file):
        unit_load_file = needed_force_file
    else:
        unit_load_file = generate_unit_nodal_force_file(structure_model.n_nodes, node_id, load_direction, 1.0)

    static_analysis = create_static_analysis_custom(structure_model, unit_load_file)
    static_analysis.solve()

    influence = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[response]]

    if load_direction in ['a','b','g'] and node_id == 0 and load_direction == GD.RESPONSE_DIRECTION_MAP[response]:
        if node_id >= response_node_id:
            return 1.0
        else:
            return 0.0
    # maybe due to numerical stuff or whatever
    # set small values that are mechanically expected to be 0 to actual 0 that in the b_sl calculation 0 and not a radnom value occurs
    if abs(influence[0]) < 1e-05:
        influence[0] = 0.0
        
    return influence[response_node_id]

def get_analytic_influences(structure_model, load_direction, node_id, response, response_node_id=0):
    '''
    for a lever arm this is simple
    if shear response -> return 1
    if base moment -> return level* 1
    '''
    moment_load = {'y':'Mz', 'z':'My', 'a':'Mx','b':'My', 'g':'Mz'}
    shear_load = {'y':'Qy', 'z':'Qz'}

    nodal_coordinates = structure_model.nodal_coordinates['x0']

    if load_direction == 'y':
        if moment_load[load_direction] == response:
            # positive
            if node_id - response_node_id <= 0:
                return 0.0
            else:
                return nodal_coordinates[node_id - response_node_id]

        elif shear_load[load_direction] == response:
            if node_id >= response_node_id:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    elif load_direction == 'z':
        if moment_load[load_direction] == response:
            # negative
            if node_id - response_node_id <= 0:
                return 0.0
            else:
                return -nodal_coordinates[node_id - response_node_id]
        elif shear_load[load_direction] == response:
            if node_id >= response_node_id:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    elif load_direction == 'x':
        return 0.0

    elif load_direction in ['a','b','g']:
        unit = '[Nm]'
        if moment_load[load_direction] == response:
            if node_id >= response_node_id:
                return 1.0
            else:
                return 0.0
        else: # moments don't cause shear forces
            return 0.0

def integrate_influence(influence_function, nodal_z):
    ''' 
    calcuate the integratl of the inflence function.
    must be given resposne and direction specific -> is a vector of values along the height
    nodal_coordinates are the x positions of the nodes (structure_model.nodal_coordinates['x0'])
    '''
    
    inf = influence_function[:-1] + influence_function[1:]
    l = nodal_z[1:] - nodal_z[:-1]

    integral = sum(np.multiply(l, inf) * 0.5)
    
    return integral

def create_static_analysis_custom(structure_model, load_vector_file):
    '''
    give the static analysis object parameters with a given load file
    '''
    analysis_params_custom= {
                "type" : "static_analysis",
                "settings": {},
                "input":{
                    "help":"provide load file in the required format - either some symbolic generated or time step from dynamic",
                    "file_path": load_vector_file,
                    "is_time_history_file" : False,
                    "selected_time_step" : 15000
                },
                "output":{
                    "plot": ["deformation", "forces"],
                    "write": ["deformation"]
                }
            }
    # use static analysis or seperate influence computation
    from source.analysis.static_analysis import StaticAnalysis
    static_analysis = StaticAnalysis(structure_model, analysis_params_custom)

    return static_analysis

def create_dynamic_analysis_custom(structure_model, dynamic_load_file):
    '''
    creates a dynamic analysis object with given load file
    TODO: include a saving and reading of objects to not always create and run an dynamic analysis
    '''
    from source.analysis.dynamic_analysis import DynamicAnalysis

    false = False
    true = True

    load = np.load(dynamic_load_file)

    dt = 0.02
    T = (load.shape[1] -1) * dt

    analysis_params_custom = {
                "type" : "dynamic_analysis",
                "settings": {
                    "solver_type": "Linear",
                    "run_in_modal_coordinates": false,
                    "time":{
                        "integration_scheme": "GenAlpha",
                        "start": 0.0,
                        "end": T,
                        "step" : dt},
                    "intial_conditions": {
                        "displacement": "None",
                        "velocity": "None",
                        "acceleration" : "None"
                    }},
                "input": {
                    "help":"provide load file in the required format",
                    "file_path": dynamic_load_file
                },
                "output":{
                    "selected_instance": {
                        "plot_step": [1500, 2361],
                        "write_step": [3276],
                        "plot_time": [30.5, 315.25],
                        "write_time": [450.15]
                    },
                    "animate_time_history" : false,
                    "animate_skin_model_time_history": false,
                    "kinetic_energy": {
                        "write": false,
                        "plot": false
                    },
                    "skin_model_animation_parameters":{
                        "start_record": 160,
                        "end_record": 200,
                        "record_step": 10
                    },
                    "selected_dof": {
                        "dof_list": [1, 2, 0, 4, 5, 3,
                                    -5,
                                    -4,
                                    -2,
                                    -1],
                        "help": "result type can be a list containing: reaction, ext_force, displacement, velocity, acceleration",
                        "result_type": [["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"],
                                        ["displacement", "velocity", "acceleration"],
                                        ["displacement", "velocity", "acceleration"],
                                        ["displacement", "velocity", "acceleration"],
                                        ["displacement", "velocity", "acceleration"]],
                        "plot_result": [[true], [true], [true], [true], [true], [true],
                                        [true, true, true],
                                        [true, true, true],
                                        [true, false, true],
                                        [true, false, true]],
                        "write_result": [[false],[false],[false],[true],[true],[true],
                                            [true, false, true],
                                            [true, false, true],
                                            [true, false, true],
                                            [true, false, true]]
                    }
                }
            }

    dynamic_analysis = DynamicAnalysis(structure_model, analysis_params_custom)

    return dynamic_analysis

def check_and_flip_sign(mode_shape_array, mode_id = None):
    '''
    change the sign of the mode shape such that the first entry is positive
    the translational dof is taken: the rotations are coupled and thus the sign is coupled
    y+ ->
    '''
    trans_rot = {'y':4,'z':2}
    flips = []
    for dof, label in enumerate(['x','y','z','a']):

        step = GD.DOFS_PER_NODE['3D']
        dof_id = GD.DOF_LABELS['3D'].index(label)

        dof_val0 = mode_shape_array[6+dof_id]
        if dof_val0 < 0:
            if mode_id < 3:
                flips.append(label)
            dof_shape = mode_shape_array[step+dof_id::step]
            mode_shape_array[step+dof_id::step] *=-1
            if label in ['y','z']:
                rot_id = dof_id + trans_rot[label]
                rot_shape = mode_shape_array[step+rot_id::step]
                mode_shape_array[step+rot_id::step] *= -1

    if mode_id < 3:
        print ('  in mode', mode_id, 'flipped',flips)
    return mode_shape_array

def get_sign_for_background(mean_load, direction):
    ''' 
    find a sign for the backgorund componetne according to the mean load distribution
    This is relevant if the mean value is small and changes sign along the height. If the background sign is choosen at each point to amplify the respective mean,
    this creates a favourable situation. 
    However for the nodal moment signals it makes sense to secied it node wise especially concerning the bottom and top node
    '''
    if direction in ['b','g']:
        background_sign = np.ones(mean_load.size)
        for node in range(len(mean_load)):
            if mean_load[node] <0:
                background_sign[node] *= -1
        return background_sign

    if all(item >= 0 for item in mean_load):
        return 1
    if all(item < 0 for item in mean_load):
        return -1
    else:
        negative_sum = sum(mean_load[mean_load < 0])
        positive_sum = sum(mean_load[mean_load > 0])
        if abs(negative_sum) > abs(positive_sum):
            return -1
        else:
            return 1

def get_radi_of_gyration(structure_model):
    ''' 
    radius of gyration is an equivalent to the Flächenträgheitsmoment 
    https://www.springer.com/de/book/9783642409660 (e.g.)
    from GL 4.7
    TODO this is for constant cross section along the height only sofar -> can easyli be extended and returned as an array or list
    ''' 
    # the 1 is added since everything is carried out at the nodes -> to get shape consitens
    Iy = [1.0]#structure_model.parameters['intervals'][0]['c_iy']
    Iz = [1.0]#structure_model.parameters['intervals'][0]['c_iz']
    Ip = [1.0]#structure_model.parameters['intervals'][0]['c_ip']

    A = [1.0]#structure_model.parameters['intervals'][0]['a']

    for e in structure_model.elements:
        Iy.append(e.Iy)
        Iz.append(e.Iz)
        Ip.append(e.Ip)
        A.append(e.A)

    # for a rectangle this can be simplified ( see GL 4.8e in Technische Mechanik)
    lz = structure_model.parameters['intervals'][0]['c_lz']
    ly = structure_model.parameters['intervals'][0]['c_ly']
    
    ry = np.sqrt(np.asarray(Iy)/np.asarray(A))
    rz = np.sqrt(np.asarray(Iz)/np.asarray(A))
    rp = np.sqrt(np.asarray(Ip)/np.asarray(A))


    radi = {'a':rp, 'b':ry, 'g':rz, 'x':1.0,'y':1.0,'z':1.0}

    return radi

def sort_row_vectors_dof_wise(unsorted_vector):
    '''
    unsorted vector is of dimenosn n_nodes * n_dofs
    sort it in to a dict with dof lables as keys
    '''
    sorted_dict = {}
    for idx, label in zip(list(range(GD.DOFS_PER_NODE['3D'])),
                            GD.DOF_LABELS['3D']):
        start = idx
        step = GD.DOFS_PER_NODE['3D']
        stop = unsorted_vector.shape[0] + idx - step
        sorted_dict[label] = unsorted_vector[start:stop+1:step]
    
    return sorted_dict

# # DYNAMIC ANALYSIS
def get_fft(given_series, sampling_freq):
    '''
    The function get_fft estimates the Fast Fourier transform of the given signal 
    sampling_freq = 1/dt
    '''

    signal_length=len(given_series)

    freq_half =  np.arange(0, 
                           sampling_freq/2 - sampling_freq/signal_length + sampling_freq/signal_length, 
                           sampling_freq/signal_length)

    # single sided fourier
    series_fft = np.fft.fft(given_series)
    series_fft = np.abs(series_fft[0:int(np.floor(signal_length/2))])/np.floor(signal_length/2)  
    
    max_length = len(freq_half)
    if max_length < len(series_fft):
        max_length = len(series_fft)
    
    freq_half = freq_half[:max_length-1]
    series_fft = series_fft[:max_length-1]
    
    return freq_half, series_fft

def extreme_value_analysis(dynamic_analysis_solved, response, type_of_return = 'estimate', P1 = 0.98):
    ''' 
    dynamic_analysi_solved: dynamic_analysis.solver object
    response: label given as dof_label, if given as response label it is convertedd
    type_of_return: wheter the estimated or the quantile value of P1 is returned (both are computed)
    ''' 
    if response in GD.RESPONSE_DIRECTION_MAP.keys():
        response = GD.RESPONSE_DIRECTION_MAP[response]

    response_id = GD.DOF_LABELS['3D'].index(response)
    dynamic_response = dynamic_analysis_solved.dynamic_reaction[response_id]

    dt = dynamic_analysis_solved.dt
    T_series = dynamic_analysis_solved.dt * len(dynamic_response)
    dur_ratio = 600 / T_series
    # # MAXMINEST NIST
    #P1 = 0.98
    max_qnt, min_qnt, max_est, min_est, max_std, min_std, Nupcross = stats_utils.maxmin_qnt_est(dynamic_response, 
                                                                        cdf_p_max = P1 , cdf_p_min = 0.0001, cdf_qnt = P1, dur_ratio = dur_ratio)
    
    abs_max_est = max([abs(max_est[0][0]), abs(min_est[0][0])])
    abs_max_qnt = max([abs(max_qnt[0]), abs(min_qnt[0])])

    if type_of_return == 'estimate':
        extreme_response = abs_max_est
    elif type_of_return == 'quantile':
        extreme_response = abs_max_qnt
    
    glob_max = max(abs(dynamic_response))

    print ('\nEstimated absolute maximum of', response)
    print ('   ', round(abs_max_est/glob_max,2) , 'of the gobal maximum')
    print ('Estimated absolute quantile value of', response)
    print ('   ', round(abs_max_qnt/glob_max,2) , 'of the gobal maximum')

    return extreme_response

def extreme_value_analysis_nist(given_series, dt, response_label, type_of_return = 'estimate', P1 = 0.98):
    ''' 
    dynamic_analysi_solved: dynamic_analysis.solver object
    response: label given as dof_label, if given as response label it is convertedd
    type_of_return: wheter the estimated or the quantile value of P1 is returned (both are computed)
    ''' 

    T_series = dt * len(given_series)
    dur_ratio = 600 / T_series
    # # MAXMINEST NIST
    #P1 = 0.98
    max_qnt, min_qnt, max_est, min_est, max_std, min_std, Nupcross = stats_utils.maxmin_qnt_est(given_series	, 
                                                                        cdf_p_max = P1 , cdf_p_min = 0.0001, cdf_qnt = P1, dur_ratio = dur_ratio)
    
    abs_max_est = max([abs(max_est[0][0]), abs(min_est[0][0])])
    abs_max_qnt = max([abs(max_qnt[0]), abs(min_qnt[0])])

    if type_of_return == 'estimate':
        extreme_response = abs_max_est
    elif type_of_return == 'quantile':
        extreme_response = abs_max_qnt
    
    glob_max = max(abs(given_series))

    print ('\nEstimated absolute maximum of', response_label)
    print ('   ', round(abs_max_est/glob_max,2) , 'of the gobal maximum')
    print ('Estimated absolute quantile value of', response_label)
    print ('   ', round(abs_max_qnt/glob_max,2) , 'of the gobal maximum')
    print ('    Upcrossing rate of 100:', Nupcross/100)
    print ('    len series:', len(given_series))
    print ('    physical time:', len(given_series) * dt)

    return abs_max_qnt, abs_max_est

def save_dynamic_results(responses = ['My','Mz','Mx','Qz','Qy']):
    ''' 
    from the dynamic analysis in pkl format it extracts the time histories and makes a .npy file for each response 
    (so far this is for moment responses at the base)
    parameters of whcih analysis is used are strongly related to the file naming!
    The resulting time series are aimed to be analysed within the max_min_est script 
    '''
    src = {2.5:os_join(*['source','ESWL','output']), 1.0:os_join(*['source','ESWL','output','damping_001'])}
    dest = os_join(*['C:\\Users','Johannes','Documents','TUM','5.MasterThesis','ExtremeValueAnalysis','input_signals'])
    n_nodes = '11'
    src = src[2.5]


    with open(src + os_sep + 'dyn_'+n_nodes +'_nodes'+'.pkl', 'rb') as dynamic_analysis_input:
        dynamic_analysis = pkl.load(dynamic_analysis_input)

    dynamic_analysis_solved = dynamic_analysis.solver

    #for response in ['My','Mz','Mx']:
    for response in responses:#,'Mx']:
    #if response in GD.RESPONSE_DIRECTION_MAP.keys():
        resp_label = response
        response = GD.RESPONSE_DIRECTION_MAP[response]

        response_id = GD.DOF_LABELS['3D'].index(response)
        dynamic_response = dynamic_analysis_solved.dynamic_reaction[response_id]
        name = 'kratos_dyn_' + resp_label 
        np.save(dest + os_sep + name, dynamic_response)
        print('\nsaved:',dest + os_sep + name)

if __name__ == '__main__':
    save_dynamic_results(rotate = True)


