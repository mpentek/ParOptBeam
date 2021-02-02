import numpy as np
#import source.auxiliary.global_definitions as GD
import os


DOFS_PER_NODE = {'2D': 3,
                 '3D': 6}

DOF_LABELS = {'2D': ['x', 'y', 'g'],
              '3D': ['x', 'y', 'z', 'a', 'b', 'g']}


def generate_static_force_file(number_of_nodes, node_of_load_application, force_direction, magnitude):
    '''
    creating force files 
    mainly for a single tip load with specified direction and magnitude
    '''
    src_path = 'input\\force\\generic_building\\'
    is_time_history = False
    number_of_time_steps = 1000
    domain_size = '3D'

    #force_direction = 'y'
    loaded_dof = (node_of_load_application -1)*DOFS_PER_NODE[domain_size] + DOF_LABELS[domain_size].index(force_direction)
    # one array for each dof at each node

    if is_time_history:
        force_data = np.zeros((DOFS_PER_NODE[domain_size]*number_of_nodes, number_of_time_steps))
        # force_data[i,j] += ...
    elif not is_time_history:
        force_data = np.zeros(DOFS_PER_NODE[domain_size]*number_of_nodes)
        force_data[loaded_dof] += magnitude

    force_file_name = src_path + 'static_force_' + str(number_of_nodes) + '_nodes_at_' + str(node_of_load_application) + '_in_' + force_direction+'.npy'
    np.save(force_file_name, force_data)
    print ('\ngenerated an simple static force: ')
    print (force_file_name)
    print ()
    return force_file_name

generate_static_force_file(20,20, 'y', magnitude=1.0e+07)