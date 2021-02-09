from math import ceil, log10
import sys
import numpy as np
import os

import source.auxiliary.global_definitions as GD
# TODO: clean up these function, see how to make the shear beam / additional rotational stiffness

CUST_MAGNITUDE = 4


def magnitude(x):
    # NOTE: ceil is supposed to be correct for positive values
    return int(ceil(log10(x)))


def map_lin_to_log(val, base=10**3):
    # it has to be defined with a min=0.0 and max=1.0
    # TODO: implment check
    return base**(1.0 - val)

def cm2inch(val_in_cm): return val_in_cm*0.3937007874

def shift_normalize(val, base=10**3):
    # TODO: implment check
    # it has to be defined with min=0.0
    shift_val = map_lin_to_log(0.0, base)
    val -= shift_val
    # it has to be defined with max=1.0
    norm_val = map_lin_to_log(1.0, base) - shift_val
    val /= norm_val
    return val


# def shape_function_lin(val): return 1-val


# TODO: try to figure out a good relationship between exp and magnitude_difference
def shape_function_exp(val, exp=CUST_MAGNITUDE): return (1-val)**exp


def evaluate_polynomial(x, coefs):
    val = 0.0
    for idx, coef in enumerate(coefs):
        val += coef * x**idx
    return val

def get_fitted_array (x, y, degree):
                
    # returns the fitted polynomial and the discrete array of displacements
    current_polynomial = np.poly1d(np.polyfit(x,y,degree))
    values = []
    for x_i in x:# evaluate the fitted eigenmode at certain intervals
        values.append(current_polynomial(x_i))
    eigenmodes_fitted = np.asarray(values)

    return eigenmodes_fitted 

def parse_load_signal(signal_raw, dofs_per_node, discard_time = None):
    '''
    sorts the load signals in a dictionary
    deletes first entries until discard_time
    '''
    if dofs_per_node != 6:
        raise Exception('load signal parsing only for 6 dofs per node - check dynamic load files')
    else:
        signal = {}
        for i, label in enumerate(GD.DOF_LABELS['3D']):
            if not discard_time:
                signal[label] = signal_raw[i::dofs_per_node]
            else:
                signal[label] = signal_raw[i::dofs_per_node]
                signal[label] = np.delete(signal[label], np.arange(0,discard_time), 1)
  
    return signal

def generate_static_force_file(number_of_nodes, node_of_load_application, force_direction, magnitude):
    '''
    creating force files 
    mainly for a single point load with specified direction and magnitude
    '''
    src_path = 'input/force/generic_building/unit_loads/'
    is_time_history = False
    number_of_time_steps = 1000
    domain_size = '3D'

    #force_direction = 'y'
    loaded_dof = (node_of_load_application)*GD.DOFS_PER_NODE[domain_size] + GD.DOF_LABELS[domain_size].index(force_direction)
    # one array for each GD.dof at each node

    if is_time_history:
        force_data = np.zeros((GD.DOFS_PER_NODE[domain_size]*number_of_nodes, number_of_time_steps))
        # force_data[i,j] += ...
    elif not is_time_history:
        force_data = np.zeros(GD.DOFS_PER_NODE[domain_size]*number_of_nodes)
        force_data[loaded_dof] += magnitude

    force_file_name = src_path + 'static_force_' + str(number_of_nodes) + '_nodes_at_' + str(node_of_load_application) + '_in_' + force_direction+'.npy'
    np.save(force_file_name, force_data)
    print ('\ngenerated an simple static force: ')
    print (force_file_name)
    print ()
    return force_file_name

def get_influence(structure_model, load_direction, node_id, response):
    '''
    influence function representing the response R due to a unit load acting at elevation z along load direction s
    '''
    src_path = 'input/force/generic_building/unit_loads/'

    needed_force_file = src_path + 'static_force_' + str(structure_model.n_nodes) + \
                        '_nodes_at_' + str(node_id) + \
                        '_in_' + load_direction+'.npy'

    if os.path.isfile(needed_force_file):
        unit_load_file = needed_force_file
    else:
        unit_load_file = generate_static_force_file(structure_model.n_nodes, node_id, load_direction, 1.0)

    analysis_params_custom= {
                "type" : "static_analysis",
                "settings": {},
                "input":{
                    "help":"provide load file in the required format - either some symbolic generated or time step from dynamic",
                    "file_path": unit_load_file,
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
    static_analysis.solve()

    influence = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[response]]

    # remove the force file that it doesn't get to much files here 
    #os.remove(unit_load_file)
    # returning here the influence of 0 since this is the reaction at the base node

    return influence[0]


def get_influence_analytically(structure_model, load_direction, level, response):
    # if shear response -> return 1
    # if base moment -> return level* 1
    if response in ['Qy', 'Qz']: #shear force
        return 1.0
    elif response in ['My', 'Mz']: # bending moment
        return level # just the lever arm
    elif response in ['Mx']:
        # depends on where the load is acting 
        # assuming in the geometric centre is that wind is distributed homogenously along the width 
        if load_direction == 'y':
            c_e = 'c_ez'
        elif load_direction == 'z':
            c_e = 'c_ey'
        else:
            raise Exception('influence of ' + load_direction + ' on the torsion is not implemented') 
        interval = structure_model.nodal_coordiantes['x0'].index(level)
        excentricity = structure_model.parameters["intervals"][interval][c_e] 
        return excentricity # e* unit load
    else:
        raise Exception('no influence function for ' + response)