import numpy as np

old = np.load("input/force/generic_building/dynamic_force_11_nodes.npy")
new = old[:,2250:]
np.save("input/force/generic_building/dynamic_force_11_nodes_new.npy", new)
r = np.load('C:\\Users\\Johannes\\Documents\\TUM\\5.MasterThesis\\ParOptBeam\\source\\ESWL\\output\\dynamic_reactions\\kratos_dyn_My.npy')
print()
