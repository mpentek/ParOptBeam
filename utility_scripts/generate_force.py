#===============================================================================
'''
Description: Create forces
'''
#===============================================================================

import os
import numpy as np


# ===============================================================
# USER INPUT

start_time = 0.0
end_time = 10.0
delta_time = 0.1
time_steps = int(end_time/delta_time) + 1

n_of_nodes = 4
dofs_per_node = 6

# ===============================================================
# FUNCTION DEFINITION

time_series = np.linspace(start_time, end_time, time_steps, endpoint=True)
np.save('array_time', time_series)

fx = np.multiply(np.ones(n_of_nodes), 20000.0)
fy = np.multiply(np.ones(n_of_nodes), 30000.0)
fz = np.multiply(np.ones(n_of_nodes), 40000.0)

mx = np.multiply(np.ones(n_of_nodes), 500000.0)
my = np.multiply(np.ones(n_of_nodes), 600000.0)
mz = np.multiply(np.ones(n_of_nodes), 700000.0)

# ===============================================================
# SAVE LOADS IN *.NPY format

dynamic_force = np.zeros([n_of_nodes * dofs_per_node, len(time_series)])

for i in range(n_of_nodes):
    # 0 = fx
    dynamic_force[i * dofs_per_node + 0, :] = np.multiply(np.ones(len(time_series)), fx[i])
    # 1 = fy
    dynamic_force[i * dofs_per_node + 1, :] = np.multiply(np.ones(len(time_series)), fy[i])
    # 2 = fz
    dynamic_force[i * dofs_per_node + 2, :] = np.multiply(np.ones(len(time_series)), fz[i])

    dynamic_force[i * dofs_per_node + 3, :] = np.multiply(np.ones(len(time_series)), mx[i])
    # 4 = my
    dynamic_force[i * dofs_per_node + 4, :] = np.multiply(np.ones(len(time_series)), my[i])
    # 5 = mz
    dynamic_force[i * dofs_per_node + 5, :] = np.multiply(np.ones(len(time_series)), mz[i])

np.save('force_dynamic_' + str(n_of_nodes) + '_nodes', dynamic_force)

static_force = np.zeros([n_of_nodes * dofs_per_node,1])

for i in range(n_of_nodes):
    # 0 = fx
    static_force[i * dofs_per_node + 0, :] = fx[i]
    # 1 = fy
    static_force[i * dofs_per_node + 1, :] = fy[i]
    # 2 = fz
    static_force[i * dofs_per_node + 2, :] = fz[i]
    # 3 = mx
    static_force[i * dofs_per_node + 3, :] = mx[i]
    # 4 = my
    static_force[i * dofs_per_node + 4, :] = my[i]
    # 5 = mz
    static_force[i * dofs_per_node + 5, :] = mz[i]

np.save('force_static_' + str(n_of_nodes) + '_nodes', static_force)

