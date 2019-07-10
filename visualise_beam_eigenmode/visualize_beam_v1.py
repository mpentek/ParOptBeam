import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import *
from matplotlib import animation


fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d', animation = True)
ax = Axes3D(fig)
fig.set_size_inches(5, 10, forward=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('height')

class Structure:
    def __init__(self, eigenmode, eigenmode_sampling_size, n_floors, floor_geometry, floor_height, densify=False):
        self.n_floors = n_floors
        self.eigenmode_x = eigenmode["x"]
        self.eigenmode_y = eigenmode["y"]
        self.eigenmode_theta = eigenmode["theta"]
        self.eigenmode_sampling_size = eigenmode_sampling_size
        self.floors = []
        self.floor_height = floor_height
        self.height = self.n_floors * self.floor_height
        self.floor_geometry = floor_geometry

        if densify:
            self._densify_contour()

    def _densify_contour(self, parts=5):
        new_floor_geometry = {}
        new_floor_geometry["points"] = []
        for i in range(len(self.floor_geometry["points"])):
            new_floor_geometry["points"].append(self.floor_geometry["points"][i])

            new_floor_geometry["points"] += self._get_equidistant_points(
                    [self.floor_geometry["points"][i % len(self.floor_geometry["points"])]["x"],
                    self.floor_geometry["points"][i % len(self.floor_geometry["points"])]["y"]],
                    [self.floor_geometry["points"][(i+1) % len(self.floor_geometry["points"])]["x"],
                    self.floor_geometry["points"][(i+1) % len(self.floor_geometry["points"])]["y"]],
                    parts)

        self.floor_geometry["points"] = new_floor_geometry["points"]

    def _get_equidistant_points(self, p1, p2, parts):
        return [{"x": val1, "y": val2} for val1, val2 in zip(np.linspace(p1[0], p2[0], parts+1),
                                                             np.linspace(p1[1], p2[1], parts+1))]

    def _create_all_floors(self, t):
        self.floors = []
        eig_z = np.linspace(0, self.height, self.eigenmode_sampling_size) # the z vector for eigenvalue
        eigenmode_x = []
        eigenmode_y = []
        eigenmode_theta = []
        for z in eig_z:
           eig_x = self.eigenmode_x(t, z)
           eigenmode_x.append( eig_x )
           eig_y = self.eigenmode_y(t, z)
           eigenmode_y.append( eig_y )
           eig_theta = self.eigenmode_theta(t, z)
           eigenmode_theta.append( eig_theta )

        for i in range(0, self.n_floors + 1):
            displacement_x = np.interp(i*self.floor_height, eig_z, eigenmode_x)
            displacement_y = np.interp(i*self.floor_height, eig_z, eigenmode_y)
            theta_z = np.interp(i*self.floor_height, eig_z, eigenmode_theta)
            self._create_single_floor(i, displacement_x, displacement_y, theta_z)


    def _create_single_floor(self, number_of_floor, displacement_x, displacement_y, theta_z):
        """
        creating single floor based on the given floor geometry
        """
        floor = []
        z = number_of_floor * self.floor_height

        for point in self.floor_geometry["points"]:
            x = point["x"] +  displacement_x
            y = point["y"] +  displacement_y
            x, y = self._apply_torsion(x, y, theta_z)
            node = Node(x, y, z)
            floor.append(node)
        self.floors.append(floor)


    def _apply_torsion(self, x0, y0, theta_z):
        x = cos(theta_z) * x0 - sin(theta_z) * y0
        y = sin(theta_z) * x0 + cos(theta_z) * y0
        return x, y


    def plot_1d_structure(self, t):
        """
        plotting the eigenmode function
        """
        global fig, ax
        x, y, z  = [], [], []
        for i in range(0, self.eigenmode_sampling_size):
            # z disctrized in the same freqeuncy as the given eigenmode sampling size
            _z = i * self.height / ( self.eigenmode_sampling_size)
            x.append(self.eigenmode_x(t, _z))
            y.append(self.eigenmode_y(t, _z))
            z.append( _z )
        z = np.array([z, z])

        ax.plot_wireframe(x, y, z, color='bk')


    def plot_frame(self):
        global fig, ax

        frames = [[] for i in range(len(self.floor_geometry["points"]))]
        for floor in self.floors:
            frame = []
            for i in range(0, len(floor)):
                frames[i].append(floor[i])

        for frame in frames:
            x, y, z  = [], [], []
            for i in range(0, len(frame)):
                x.append(frame[i].x)
                y.append(frame[i].y)
                z.append(frame[i].z)

            z = np.array([z, z])
            ax.plot_wireframe(x, y, z)


    def plot_all_floors(self):
        global fig, ax

        for floor in self.floors:
            x, y, z  = [], [], []
            for node in floor:
                x.append(node.x)
                y.append(node.y)
                z.append(node.z)
            x.append(floor[0].x)
            y.append(floor[0].y)
            z.append(floor[0].z)

            z = np.array([z, z])
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.view_init(10, 45)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.plot_wireframe(x, y, z)


    def update(self, t):
        """
        ploting the deformed structure with respect to the eigenmode
        """
        global fig, ax

        wframe = fig.gca()
        if wframe!=None:
            ax.cla()

        self._create_all_floors(t)
        self.plot_all_floors()
        self.plot_frame()
        self.plot_1d_structure(t)


    def animate(self):
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        a = animation.FuncAnimation(fig, self.update, 20, repeat = False)
        a.save("structure_eigenmode.avi")
        plt.show()


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


if __name__ == "__main__":
    floor_geometry = {
        "points": [{
            "x": 5.0,
            "y": 5.0
        },{
            "x": 5.0,
            "y": 0.0
        },{
            "x": 10.0,
            "y": 0.0
        },{
            "x": 10.0,
            "y": -5.0
        },{
            "x": -10.0,
            "y": -5.0
        },{
            "x": -10.0,
            "y": 0.0
        },{
            "x": -5.0,
            "y": 0.0
        },{
            "x": -5.0,
            "y": 5.0
        }]
    }
    eigenmode_sampling_size = 40
    eigenmode = {
        "x": lambda t, z: sin(0.05*z)*(2*cos(t)+sin(t)),
        "y": lambda t, z: 2*sin(0.05*z)*(4*cos(t)+3*sin(t)),
        "theta" : lambda t, z: 0.2*sin(0.05*z)*(2*cos(t)+sin(t))
    }
    eigenmode_pure_torsion = {
        "x": lambda t, z: t/(t+1e-32) * z/(z+1e-32),
        "y": lambda t, z: t/(t+1e-32) * z/(z+1e-32),
        "theta" : lambda t, z: 0.2*sin(0.05*z)*(2*cos(t)+sin(t))
    }
    myStructure = Structure(eigenmode, eigenmode_sampling_size, 20, floor_geometry, 4)
    myStructure.animate()