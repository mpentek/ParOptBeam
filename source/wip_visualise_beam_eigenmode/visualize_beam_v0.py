import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Structure:
    def __init__(self, eigenmode, n_floors, floor_geometry ,floor_height):
        self.n_floors = n_floors
        self.eigenmode = eigenmode
        self.floors = []
        self.floor_height = floor_height
        self.height = self.n_floors * self.floor_height
        self.floor_geometry = floor_geometry


    def create_all_floors(self):
        eig_z = np.linspace(0, self.height, len(self.eigenmode))

        for i in range(0, self.n_floors+1):
            magnitude_x = np.interp(i*self.floor_height, eig_z, self.eigenmode)
            magnitude_y = 0
            self.create_single_floor(i, magnitude_x, magnitude_y)


    def create_single_floor(self, number_of_floor, magnitude_x, magnitude_y):
        """
        creating single floor based on the given floor geometry
        """
        floor = []
        z = number_of_floor * self.floor_height

        for point in self.floor_geometry["points"]:
            x = point["x"] +  magnitude_x
            y = point["y"] +  magnitude_y
            node = Node(x, y, z)
            floor.append(node)
        self.floors.append(floor)


    def plot_1d_structure(self, axis):
        """
        plotting the eigenmode function
        """
        x, y, z  = [], [], []
        for i in range(0, len(self.eigenmode)):
            x.append(self.eigenmode[i])
            y.append(0.0)
            z.append( i * self.height / ( len( self.eigenmode ) - 1) )
        z = np.array([z, z])
        axis.plot_wireframe(x, y, z, color='bk')


    def plot_structure(self):
        """
        ploting the deformed structure with respect to the eigenmode
        """
        figure = plt.figure()
        axis = figure.add_subplot(111, projection = '3d')
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
            axis.plot_wireframe(x, y, z)

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
            axis.plot_wireframe(x, y, z)

        self.plot_1d_structure(axis)

        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_zlabel('z')

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
    eig_z = np.linspace(0, 4*np.pi, 20)
    eigenmode = np.sin(0.5*eig_z)
    myStructure = Structure(eigenmode, 40, floor_geometry, 4)
    myStructure.create_all_floors()
    myStructure.plot_structure()