import numpy as np
import os
from os.path import sep as os_sep
from os.path import join as os_join

import source.auxiliary.global_definitions as GD
import source.auxiliary.statistics_utilities as stats_utils

# # ========= ESWL =============


def parse_load_signal(signal_raw, time_info,dofs_per_node,  load_directions_to_include = 'all', discard_time = None):
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

    if isinstance(time_info, np.ndarray):
        fs = 1/(time_array[1] - time_array[0]) # simulation time step
    elif time_info < 10.0:
        print('CHECK: Time Info is TIME STEP?')
        fs = 1/time_info
    else:
        print('CHECK: Time Info is SAMPLE FREQUENCY?')
        fs = time_info

    signal['sample_freq'] = fs

    return signal

def parse_eswl_components(eswl_dict, component = 'total'):
    '''
    makes a load vector that can be used in a static analysis 
    '''

    shape = GD.DOFS_PER_NODE['3D'] * len(eswl_dict[list(eswl_dict.keys())[0]][component])
    load_vector = np.zeros(shape)
    for label in eswl_dict:
        dof_label_id = GD.DOF_LABELS['3D'].index(label)
        for i, val in enumerate(eswl_dict[label][component]):
            sort_id = i * GD.DOFS_PER_NODE['3D'] + dof_label_id
            load_vector[sort_id] = val

    #TODO: must this be transposed?!
    return load_vector


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
        load = parse_eswl_components(load_vector)
    else:
        load = load_vector

    src_path = os_join('input','force','generic_building','eswl')
    if not os.path.isdir(src_path):
        os.makedirs(src_path)

    file_name = os_join(src_path, 'eswl_' + str(int(len(load)/GD.DOFS_PER_NODE['3D'])) + '_nodes.npy')

    np.save(file_name, load)

    return file_name


def generate_unit_nodal_force_file(number_of_nodes, node_of_load_application, force_direction, magnitude):
    '''
    creating a force .npy file with a nodal force at given node, direction and magnitude
    '''
    src_path = os.path.join(*['input','force','generic_building','unit_loads'])
    domain_size = '3D'

    loaded_dof = (node_of_load_application)*GD.DOFS_PER_NODE[domain_size] + GD.DOF_LABELS[domain_size].index(force_direction)

    force_data = np.zeros(GD.DOFS_PER_NODE[domain_size]*number_of_nodes)
    force_data[loaded_dof] = magnitude
    force_data = force_data.reshape(GD.DOFS_PER_NODE[domain_size]*number_of_nodes,1)

    force_file_name = src_path + os.path.sep + 'unit_static_force_' + str(number_of_nodes) + '_nodes_at_' + str(node_of_load_application) + \
                        '_in_' + force_direction+'.npy'
    np.save(force_file_name, force_data)

    return force_file_name


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

def extreme_value_analysis_nist(time_series, time, type_of_return = 'estimate', P1 = 0.98):
    ''' 
    Extreme value analysis using NIST TP approach
    time_series: time series
    time: dictionary (used for duration ratio calculation)
        dt: time step of time series
        T: time for which the extreme shall be estimated
    type_of_return: wheter the estimated or the quantile value of P1 is returned (both are computed)
    P1: probaility of non exceedance
    ''' 

    T_series = time['dt'] * len(time_series)
    dur_ratio = time['T'] / T_series
    # # MAXMINEST NIST
    max_qnt, min_qnt, max_est, min_est, max_std, min_std, Nupcross = stats_utils.maxmin_qnt_est(time_series	, 
                                                                        cdf_p_max = P1 , cdf_p_min = 0.0001, cdf_qnt = P1, dur_ratio = dur_ratio)
    
    abs_max_est = max([abs(max_est[0][0]), abs(min_est[0][0])])
    abs_max_qnt = max([abs(max_qnt[0]), abs(min_qnt[0])])

    if type_of_return == 'estimate':
        extreme_response = abs_max_est
    elif type_of_return == 'quantile':
        extreme_response = abs_max_qnt
    
    return extreme_response

