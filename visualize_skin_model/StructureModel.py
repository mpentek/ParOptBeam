import json
import numpy as np
from visualize_skin_model.NodeModel import Node
from sympy import Plane, Point3D

CONTOUR_DENSITY = 1


class Floor:
    def __init__(self, floor_geometry, height):
        """
        creating single floor based on the given floor geometry with the floor height
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [],[],[]
        for point in floor_geometry:
            x = point["x"]
            y = point["y"]
            z = height
            self.x_vec.append(x)
            self.y_vec.append(y)
            self.z_vec.append(z)
            node = Node(x, y, z)
            self.nodes.append(node)

        self.plane = Plane((self.nodes[0 * CONTOUR_DENSITY].x, self.nodes[0 * CONTOUR_DENSITY].y, self.nodes[0 * CONTOUR_DENSITY].z),
                           (self.nodes[1 * CONTOUR_DENSITY].x, self.nodes[1 * CONTOUR_DENSITY].y, self.nodes[1 * CONTOUR_DENSITY].z),
                           (self.nodes[2 * CONTOUR_DENSITY].x, self.nodes[2 * CONTOUR_DENSITY].y, self.nodes[2 * CONTOUR_DENSITY].z))

    def print_floor(self):
        for node in self.nodes:
            node.print_info()


class Frame:
    def __init__ (self, floors, index):
        """
        connecting all points from the same geometry point for each floor
        """ 
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [],[],[]
        for floor in floors:
            node = floor.nodes[index]
            self.nodes.append(node)
            self.x_vec.append(node.x0)
            self.y_vec.append(node.y0)
            self.z_vec.append(node.z0)


class Structure:
    def __init__(self, structure_file):
        """
        initializing structure with geometry
        """
        self.floors = []
        self.frames = []
        with open(structure_file) as json_file:
            data = json.load(json_file)
            self.floor_geometry = data["geometry"]
            self.structure_height = data["height"]
            self.num_of_floors = data["num_of_floors"]
            try:
                self.floor_height = data["floor_height"]
            except:
                self.floor_height = self.structure_height/self.num_of_floors

        self.densify_contour(CONTOUR_DENSITY)
        self.print_structure_info()
        self.create_floors()
        self.create_frames()

    def print_structure_info(self):    
        msg = "=============================================\n"
        msg += "STRUCTURE MODEL INFO \n"
        msg += "HEIGHT:\t" + str(self.structure_height) + "\n"
        msg += "#FLOOR:\t" + str(self.num_of_floors) + "\n"
        msg += "FLOOR HEIGHT:\t" + str(self.floor_height) + "\n"
        msg += "============================================="
        print(msg)

    def create_floors(self):
        current_height = 0.0
        while current_height <= self.structure_height:       
            floor = Floor(self.floor_geometry, current_height)
            self.floors.append(floor)
            current_height += self.floor_height
        if current_height !=self.structure_height:
            floor = Floor(self.floor_geometry, self.structure_height)
            self.floors.append(floor)
       
    def create_frames(self):
        for i in range(len(self.floor_geometry)):
            frame = Frame(self.floors, i)
            self.frames.append(frame)

    def densify_contour(self, parts=5):
        if parts > 1:
            new_floor_geometry = []
            for i in range(len(self.floor_geometry)):
                new_floor_geometry.append(self.floor_geometry[i])

                new_floor_geometry += self._get_equidistant_points(
                        [self.floor_geometry[i % len(self.floor_geometry)]["x"],
                         self.floor_geometry[i % len(self.floor_geometry)]["y"]],
                        [self.floor_geometry[(i+1) % len(self.floor_geometry)]["x"],
                         self.floor_geometry[(i+1) % len(self.floor_geometry)]["y"]],
                        parts)

            self.floor_geometry = new_floor_geometry

    @staticmethod
    def _get_equidistant_points(p1, p2, parts):
        return [{"x": val1, "y": val2} for val1, val2 in zip(np.linspace(p1[0], p2[0], parts+1),
                                                             np.linspace(p1[1], p2[1], parts+1))]

    def apply_transformation_for_structure(self):
        for floor in self.floors:
            for i in range(len(floor.nodes)):
                floor.nodes[i].apply_transformation()
                floor.x_vec[i] = floor.nodes[i].x
                floor.y_vec[i] = floor.nodes[i].y
                floor.z_vec[i] = floor.nodes[i].z

        for frame in self.frames:
            for i in range(len(frame.nodes)):
                frame.nodes[i].apply_transformation()
                frame.x_vec[i] = frame.nodes[i].x
                frame.y_vec[i] = frame.nodes[i].y
                frame.z_vec[i] = frame.nodes[i].z

    def print_floor(self, floor_id):
        print("Printing Floor: " + str(floor_id))
        self.floors[floor_id].print_floor()


if __name__ == "__main__":
    s = Structure("trapezoid.json")