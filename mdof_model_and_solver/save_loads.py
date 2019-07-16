# ===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18 
    Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
    Analysis type base class and derived classes specific types

Author: mate.pentek@tum.de, anoop.kodakkal@tum.de, catharina.czech@tum.de, peter.kupas@tum.de

      
Note:   UPDATE: The script has been written using publicly available information and 
    data, use accordingly. It has been written and tested with Python 2.7.9.
    Tested and works also with Python 3.4.3 (already see differences in print).
    Module dependencies (-> line 61-74): 
        python
        numpy
        sympy
        matplotlib.pyplot

Created on:  22.11.2017
Last update: 30.11.2017
'''
# ===============================================================================
import numpy as np
import matplotlib.pyplot as plt

from source.structure_model import*
from source.analysis_type import*
from source.load_type import*

import json
import sys
import time

'''
This save load functionality only applies for a certain example
the one of the bridge pylon
for which during CFD forces for 25 nodes were recorded
'''

number_of_elements = 24
number_of_nodes = number_of_elements + 1
dofs_per_node = 6

#  only applicable to fixed boundary condition 
force_data_folder = 'level_force'

array_time = np.loadtxt(force_data_folder+'/level_0.dat', usecols=(0,), skiprows=7)
dynamic_force = np.zeros([number_of_nodes * dofs_per_node, len(array_time)])

# extracting the forces from the files
for level in np.arange(number_of_elements):
    array_time_i = np.loadtxt(force_data_folder+'/level_'+str(level)+'.dat', usecols=(0,), skiprows=7)
    force_i = np.zeros([dofs_per_node,len(array_time)])
    force_from_file = np.loadtxt(force_data_folder+'/level_'+str(level)+'.dat', usecols=(1,2,3,4,5,6), skiprows=7)
    force_i[0,:] = force_from_file[:,2]
    force_i[1,:] = force_from_file[:,1]
    force_i[2,:] = force_from_file[:,0]
    force_i[3,:] = force_from_file[:,5]
    force_i[4,:] = force_from_file[:,4]
    force_i[5,:] = force_from_file[:,3]
    # TODO : check better ways tto do this 
    if np.array_equal(array_time,array_time_i): 
        dynamic_force[level*dofs_per_node:(level+1)*dofs_per_node,:] = force_i
    else:
        err_msg = "The time array doesnt match : please check the data"
        raise Exception(err_msg)
np.save('array_time',array_time)
np.save('force_dynamic',dynamic_force)        
del_time = time.time()-start_time

print(del_time)
print('Files saved')
