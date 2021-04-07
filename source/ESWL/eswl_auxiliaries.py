import numpy as np
import os

import source.auxiliary.global_definitions as GD

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

def parse_load_signal(signal_raw, dofs_per_node, discard_time = None):
    '''
    sorts the load signals in a dictionary with load direction as keys: 
    x,y,z: nodal force 
    a,b,g: nodal moments 
    deletes first entries until discard_time
    '''
    if dofs_per_node != 6:
        raise Exception('load signal parsing only for 6 dofs per node - check dynamic load files')
    else:
        signal = {}
        for i, label in enumerate(GD.DOF_LABELS['3D']):
            signal[label] = signal_raw[i::dofs_per_node]
            
            if discard_time:                
                signal[label] = np.delete(signal[label], np.arange(0,discard_time), 1)
  
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

    return signal_raw

def generate_static_load_vector_file(load_vector):
    '''
    return a npy file and saves it for a given load.

    If load is given as dictionary with directions as keys it parses it to a 1D array that is needed in the ParOptBeam
    '''
    if isinstance(load_vector, dict):
        load = parse_load_signal_backwards(load_vector)
    else:
        load = load_vector

    src_path = 'input/force/generic_building/eswl/'
    file_name = src_path + 'eswl_' + str(int(len(load)/GD.DOFS_PER_NODE['3D'])) + '_nodes.npy'

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
    src_path = 'input/force/generic_building/unit_loads/'
    domain_size = '3D'

    loaded_dof = (node_of_load_application)*GD.DOFS_PER_NODE[domain_size] + GD.DOF_LABELS[domain_size].index(force_direction)

    force_data = np.zeros(GD.DOFS_PER_NODE[domain_size]*number_of_nodes)
    force_data[loaded_dof] += magnitude
    force_data = force_data.reshape(GD.DOFS_PER_NODE[domain_size]*number_of_nodes,1)

    force_file_name = src_path + 'unit_static_force_' + str(number_of_nodes) + '_nodes_at_' + str(node_of_load_application) + \
                        '_in_' + force_direction+'.npy'
    np.save(force_file_name, force_data)
    
    return force_file_name

def get_influence(structure_model, load_direction, node_id, response):
    '''
    influence function representing the response R due to a unit load acting at elevation z along load direction s
    '''
    src_path = 'input/force/generic_building/unit_loads/'

    needed_force_file = src_path + 'unit_static_force_' + str(structure_model.n_nodes) + \
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
        return 1.0
    # maybe due to numerical stuff or whatever 
    # set small values that are mechanically expected to be 0 to actual 0 that in the b_sl calculation 0 and not a radnom value occurs
    if abs(influence[0]) < 1e-05:
        influence[0] = 0.0 
    # returning here the influence of 0 since this is the reaction at the base node
    return influence[0]

def get_decoupled_influences(structure_model, load_direction, node_id, response):
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
            return nodal_coordinates[node_id]
        elif shear_load[load_direction] == response:
            return 1.0     
        elif response == 'Mx':
            c_e = 'c_ez'
            if len(structure_model.parameters["intervals"]):
                # TODO: here only the case where intervals = n_elems is correctly implemented sofar
                # if this is not so the excentricity from the first interval is taken for all
                excentricity = structure_model.parameters["intervals"][0][c_e][0]
            else:
                if node_id == 0:
                    excentricity = structure_model.parameters["intervals"][node_id][c_e][0]
                else: 
                    excentricity = structure_model.parameters["intervals"][node_id-1][c_e][0] 
            # negative sign for positive Mx
            return -excentricity
        else:
            return 0.0

    elif load_direction == 'z':
        if moment_load[load_direction] == response:
            # negative
            return -nodal_coordinates[node_id]
        elif shear_load[load_direction] == response:
            return 1.0     
        elif response == 'Mx':
            c_e = 'c_ey'
            if len(structure_model.parameters["intervals"]):
                # TODO: here only the case where intervals = n_elems is correctly implemented sofar
                # if this is not so the excentricity from the first interval is taken for all
                excentricity = structure_model.parameters["intervals"][0][c_e][0]
            else:
                if node_id == 0:
                    excentricity = structure_model.parameters["intervals"][node_id][c_e][0]
                else: 
                    excentricity = structure_model.parameters["intervals"][node_id-1][c_e][0] 
            # positive sign for positive Mx
            return excentricity
        else:
            return 0.0

    elif load_direction == 'x':
        return 0.0

    elif load_direction in ['a','b','g']:
        unit = '[Nm]'
        if moment_load[load_direction] == response:
            return 1.0
        else: # moments don't cause shear forces
            return 0.0

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
    '''
    from source.analysis.dynamic_analysis import DynamicAnalysis

    false = False
    true = True

    analysis_params_custom = {
                "type" : "dynamic_analysis",
                "settings": {
                    "solver_type": "Linear",
                    "run_in_modal_coordinates": false,
                    "time":{
                        "integration_scheme": "GenAlpha",
                        "start": 0.0,
                        "end": 600.0,
                        "step" : 0.02},
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
        print ('in mode', mode_id, 'flipped',flips)        
    return mode_shape_array 
    