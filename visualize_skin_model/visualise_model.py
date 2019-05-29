import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from model import Structure, Floor, Node

plt.rcParams['legend.fontsize'] = 16

def visualise_structure(structure):
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection = '3d', aspect=10, azim=-40, elev=10)

    visualise_floor(structure.floors, ax)
    visualise_floor(structure.frames, ax)
    
    plt.tight_layout()
    plt.show()
    

def visualise_floor(floors, ax):
    for floor in floors:
        z = np.array([floor.z_vec, floor.z_vec])
        ax.plot_wireframe(floor.x_vec, floor.y_vec, z)


def visualize_frame(frames, ax):
    for frame in frames:
        z = np.array([frame.z_vec, frame.z_vec])
        ax.plot_wireframe(frame.x_vec, frame.y_vec, z)


def animate(self):
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    a = animation.FuncAnimation(fig, self.update, 40, repeat = False)
    #a.save("structure_displacement.avi")
    plt.show()


def update(self, t):
    """
    ploting the deformed structure with respect to the displacement
    """
    global fig, ax

    wframe = fig.gca()
    if wframe!=None:
        ax.cla()

    self._create_all_floors(t)
    self.plot_all_floors()
    self.plot_frame()


