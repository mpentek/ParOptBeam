import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from model import Structure, Floor, Node

plt.rcParams['legend.fontsize'] = 16

def visualise_structure(structure):
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection = '3d', aspect=10, azim=-40, elev=10)

    for floor in structure.floors:
        visualise_floor(floor, ax)
    
    plt.tight_layout()
    plt.show()
    
def visualise_floor(floor, ax):
    x, y, z  = [], [], []
    # adding coordinates to one vector    

    z = np.array([floor.z_vec, floor.z_vec])
    ax.plot_wireframe(floor.x_vec, floor.y_vec, z)