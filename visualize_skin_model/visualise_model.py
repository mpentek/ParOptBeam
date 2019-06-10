import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from NodeModel import Node
from StructureModel import Structure, Floor

plt.rcParams['legend.fontsize'] = 16

class Visualiser:
    def __init__(self, structure, line_structure):
        self.structure = structure
        self.line_structure = line_structure

        self.fig = plt.figure(figsize=(5, 10))
        self.ax = self.fig.add_subplot(111, projection = '3d', aspect=10, azim=-40, elev=10)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.visualise_structure()
        self.visualise_line_structure()
        self.ax.set_zlim([0, self.ax.get_zlim()[1]])      
        plt.tight_layout()
        plt.show()

    def visualise_line_structure(self):
        z = np.array([self.line_structure.z_vec, self.line_structure.z_vec])
        self.ax.plot_wireframe(self.line_structure.x_vec, self.line_structure.y_vec, z, color='k')
        for node in self.line_structure.nodes:
            self.ax.scatter(node.x0, node.y0, node.z0, marker='o', c='r', s = 500)

    def visualise_structure(self):
        self.visualise_floor()
        self.visualize_frame()

    def visualise_floor(self):
        for floor in self.structure.floors:
            z = np.array([floor.z_vec, floor.z_vec])
            self.ax.plot_wireframe(floor.x_vec, floor.y_vec, z)

    def visualize_frame(self):
        for frame in self.structure.frames:
            z = np.array([frame.z_vec, frame.z_vec])
            self.ax.plot_wireframe(frame.x_vec, frame.y_vec, z)


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


