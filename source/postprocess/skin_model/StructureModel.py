import json
import numpy as np
from os.path import join

from source.postprocess.skin_model.NodeModel import Node
from source.postprocess.skin_model.Mapper import interpolate_points


DIRECTION_VECTOR = ["x", "y", "z", "x", "y", "z"]
beam_direction_index = 0


class Element:
    def __init__(self, geometry, s, scale=1., beam_direction="x"):
        """
        creating single floor based on the given floor geometry with the floor height
        @:param s: coordinate in the beam direction
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [], [], []
        self.x0_vec, self.y0_vec, self.z0_vec = [], [], []
        self.scale = scale

        for point in geometry:
            # the beam direction takes a dummy value 0 at the beginning and will be overwritten
            x = point["x"] * scale
            y = point["y"] * scale
            z = point["z"] * scale

            if beam_direction == "x":
                x = s
                beam_direction_index = 0
            elif beam_direction == "y":
                y = s
                beam_direction_index = 1
            elif beam_direction == "z":
                z = s
                beam_direction_index = 2
            else:
                x = s
                print("Unknown beam direction, setting default beam direction to x")
            self.x0_vec.append(x)
            self.y0_vec.append(y)
            self.z0_vec.append(z)
            self.x_vec.append(x)
            self.y_vec.append(y)
            self.z_vec.append(z)
            node = Node(x, y, z)
            self.nodes.append(node)

        # assigning the beam-wise vector
        if beam_direction == "x":
            self.s_vec = self.x_vec
        elif beam_direction == "y":
            self.s_vec = self.y_vec
        elif beam_direction == "z":
            self.s_vec = self.z_vec

    def print_element(self):
        for node in self.nodes:
            node.print_info()

    def print_element_normal(self):
        # TODO
        print(self.plane.normal_vector)


class Frame:
    def __init__(self, floors, index, beam_direction="x"):
        """
        connecting all points from the same geometry point for each floor
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec, self.s_vec = [], [], [], []
        self.x0_vec, self.y0_vec, self.z0_vec = [], [], []

        for floor in floors:
            node = floor.nodes[index]
            self.nodes.append(node)
            self.x0_vec.append(node.x0)
            self.y0_vec.append(node.y0)
            self.z0_vec.append(node.z0)
            self.x_vec.append(node.x0)
            self.y_vec.append(node.y0)
            self.z_vec.append(node.z0)

        if beam_direction == "x":
            self.s_vec = self.x_vec
        elif beam_direction == "y":
            self.s_vec = self.y_vec
        elif beam_direction == "z":
            self.s_vec = self.z_vec


class Structure:
    def __init__(self, structure_file):
        """
        initializing structure with geometry
        """
        self.elements = []
        self.frames = []
        self.contour_density = 1
        with open(structure_file) as json_file:
            data = json.load(json_file)
            self.element_geometry = data["geometry"]

            # NOTE for now a workaround for importing
             self.dof_file = join(
                *["input", "deformation", data["dofs_file_name"]])

            self.beam_length = json.load(open(self.dof_file))["length"]
            self.scaling_vector = data["scaling_vector"]
            self.num_of_elements = len(self.scaling_vector)
            self.beam_direction = data["beam_direction"]
            self.element_length = self.beam_length / self.num_of_elements
            self.is_record_animation = data["record_animation"]
            self.is_visualize_line_structure = data["visualize_line_structure"]
            self.contour_density = data["contour_density"]

        self.densify_contour(self.contour_density)
        self.print_structure_info()
        self.create_elements()
        self.create_frames()

    def print_structure_info(self):
        msg = "=============================================\n"
        msg += "BEAM MODEL INFO \n"
        msg += "LENGTH:\t" + str(self.beam_length) + "\n"
        msg += "#ELEMENTS:\t" + str(self.num_of_elements) + "\n"
        msg += "ELEMENT LENGTH:\t" + str(self.element_length) + "\n"
        msg += "============================================="
        print(msg)

    def create_elements(self):
        current_length = 0.0
        current_scale = 1.
        while current_length <= self.beam_length:
            current_scale = interpolate_points(current_length, np.linspace(0, self.beam_length, self.num_of_elements),
                                               self.scaling_vector)
            element = Element(self.element_geometry, current_length,
                              current_scale, self.beam_direction)
            self.elements.append(element)
            current_length += self.element_length
        if current_length <= self.beam_length:
            element = Element(self.element_geometry, self.beam_length,
                              current_scale, self.beam_direction)
            self.elements.append(element)

    def create_frames(self):
        for i in range(len(self.element_geometry)):
            frame = Frame(self.elements, i, self.beam_direction)
            self.frames.append(frame)

    def densify_contour(self, parts=5):
        if parts > 1:
            new_floor_geometry = []
            for i in range(len(self.element_geometry)):
                new_floor_geometry.append(self.element_geometry[i])

                new_floor_geometry += self._get_equidistant_points(
                    [self.element_geometry[i % len(self.element_geometry)][DIRECTION_VECTOR[beam_direction_index + 1]],
                     self.element_geometry[i % len(self.element_geometry)][DIRECTION_VECTOR[beam_direction_index + 2]]],
                    [self.element_geometry[(i + 1) % len(self.element_geometry)][
                        DIRECTION_VECTOR[beam_direction_index + 1]],
                     self.element_geometry[(i + 1) % len(self.element_geometry)][
                         DIRECTION_VECTOR[beam_direction_index + 2]]],
                    parts)
            print(DIRECTION_VECTOR[beam_direction_index])
            self.element_geometry = new_floor_geometry

    @staticmethod
    def _get_equidistant_points(p1, p2, parts):
        return [{DIRECTION_VECTOR[beam_direction_index]: 0.0, DIRECTION_VECTOR[beam_direction_index + 1]: val1,
                 DIRECTION_VECTOR[beam_direction_index + 2]: val2} for val1, val2 in
                zip(np.linspace(p1[0], p2[0], parts + 1),
                    np.linspace(p1[1], p2[1], parts + 1))]

    def apply_transformation_for_structure(self):
        for floor in self.elements:
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

    def print_structure_element(self, floor_id):
        print("Printing Floor: " + str(floor_id))
        self.elements[floor_id].print_element()

    def print_one_structure_normal(self, floor_id):
        self.elements[floor_id].print_element_normal()


if __name__ == "__main__":
    s = Structure("trapezoid.json")
