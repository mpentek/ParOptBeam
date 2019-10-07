from typing import List

import numpy as np
from enum import IntEnum

from source.postprocess.skin_model.NodeModel import Node
from source.postprocess.skin_model.Mapper import interpolate_points

DIRECTION_VECTOR = ["x", "y", "z", "x", "y", "z"]


class BeamDirection(IntEnum):
    x = 0
    y = 1
    z = 2


class Element:
    def __init__(self, geometry, current_length, beam_direction, scale=1.):
        """
        creating single element based on the given element geometry with the element height
        """
        self.num_of_nodes = len(geometry)
        self.nodes = np.empty(self.num_of_nodes, dtype=Node)
        self.undeformed = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.deformed = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.scale = scale
        for i in range(len(geometry)):
            x = geometry[i][0]
            y = geometry[i][1]
            z = geometry[i][2]

            # Note: beam_direction won't be scaled
            position = [x, y, z]
            position *= scale
            position[int(beam_direction)] = current_length
            self.nodes[i] = Node(position)
            self.undeformed[0][i] = self.nodes[i].undeformed[0]
            self.undeformed[1][i] = self.nodes[i].undeformed[1]
            self.undeformed[2][i] = self.nodes[i].undeformed[2]

        self.deformed = self.undeformed

    def print_element(self):
        for node in self.nodes:
            node.print_info()


class Frame:
    def __init__(self, elements, index):
        """
        connecting all points from the same geometry point for each element
        """

        self.num_of_nodes = len(elements)
        self.nodes = np.empty(self.num_of_nodes, dtype=Node)
        self.undeformed = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.deformed = np.ndarray((3, self.num_of_nodes), dtype=float)

        for i in range(len(elements)):
            self.nodes[i] = elements[i].nodes[index]
            self.undeformed[0][i] = self.nodes[i].undeformed[0]
            self.undeformed[1][i] = self.nodes[i].undeformed[1]
            self.undeformed[2][i] = self.nodes[i].undeformed[2]

        self.deformed = self.undeformed

    def print_frame(self):
        for node in self.nodes:
            node.print_info()


class Structure:
    def __init__(self, params):
        """
        initializing structure with geometry
        """

        self.dofs = params["dofs_input"]
        self.element_geometry = params["geometry"]
        self.beam_length = params["length"]
        self.scaling_vector = params["scaling_vector"]
        self.num_of_frames = len(self.element_geometry)
        self.num_of_elements = len(self.scaling_vector) + 1
        self.beam_direction = BeamDirection[params["beam_direction"]]
        self.element_length = self.beam_length / (self.num_of_elements - 1)
        self.is_record_animation = params["record_animation"]
        self.is_visualize_line_structure = params["visualize_line_structure"]
        self.contour_density = params["contour_density"]
        self.print_structure_info()

        self.densify_contour(self.contour_density)
        self.elements = np.empty(self.num_of_elements, dtype=Element)
        self.create_elements()
        self.frames = np.empty(self.num_of_frames, dtype=Frame)
        self.create_frames()

    def print_structure_info(self):
        msg = "=============================================\n"
        msg += "VISUALISING SKIN MODEL"
        msg += "BEAM MODEL INFO \n"
        msg += str(self.beam_direction) + "\n"
        msg += "LENGTH:\t" + str(self.beam_length) + "\n"
        msg += "#ELEMENTS:\t" + str(self.num_of_elements) + "\n"
        msg += "ELEMENT LENGTH:\t" + str(self.element_length) + "\n"
        msg += "============================================="
        print(msg)

    def create_elements(self):
        element_vec = np.linspace(self.element_length / 2, self.beam_length - self.element_length / 2, self.num_of_elements - 1)

        for i in range(self.num_of_elements):
            current_length = i * self.element_length
            current_scale = interpolate_points(current_length, element_vec, self.scaling_vector)
            element = Element(self.element_geometry, current_length, self.beam_direction, current_scale)
            self.elements[i] = element

    def create_frames(self):
        for i in range(self.num_of_frames):
            frame = Frame(self.elements, i)
            self.frames[i] = frame

    def densify_contour(self, parts=5):
        if parts > 1:
            new_element_geometry = []
            for i in range(len(self.element_geometry)):
                new_element_geometry.append(self.element_geometry[i])

                new_element_geometry += self._get_equidistant_points(
                    [self.element_geometry[i % len(self.element_geometry)][DIRECTION_VECTOR[beam_direction_index + 1]],
                     self.element_geometry[i % len(self.element_geometry)][DIRECTION_VECTOR[beam_direction_index + 2]]],
                    [self.element_geometry[(i + 1) % len(self.element_geometry)][
                         DIRECTION_VECTOR[beam_direction_index + 1]],
                     self.element_geometry[(i + 1) % len(self.element_geometry)][
                         DIRECTION_VECTOR[beam_direction_index + 2]]],
                    parts)
            # print(DIRECTION_VECTOR[beam_direction_index])
            self.element_geometry = new_element_geometry

    @staticmethod
    def _get_equidistant_points(p1, p2, parts):
        return [{DIRECTION_VECTOR[beam_direction_index]: 0.0, DIRECTION_VECTOR[beam_direction_index + 1]: val1,
                 DIRECTION_VECTOR[beam_direction_index + 2]: val2} for val1, val2 in
                zip(np.linspace(p1[0], p2[0], parts + 1),
                    np.linspace(p1[1], p2[1], parts + 1))]

    def apply_transformation_for_structure(self):
        for e in self.elements:
            for i in range(len(e.nodes)):
                e.nodes[i].apply_transformation()
                e.deformed[0][i] = e.nodes[i].deformed[0]
                e.deformed[1][i] = e.nodes[i].deformed[1]
                e.deformed[2][i] = e.nodes[i].deformed[2]

        for f in self.frames:
            for i in range(len(f.nodes)):
                f.nodes[i].apply_transformation()
                f.deformed[0][i] = f.nodes[i].deformed[0]
                f.deformed[1][i] = f.nodes[i].deformed[1]
                f.deformed[2][i] = f.nodes[i].deformed[2]

    def print_structure_element(self, element_id):
        print("Printing element: " + str(element_id))
        self.elements[element_id].print_element()

    def print_structure_frame(self, frame_id):
        print("Printing frame: " + str(frame_id))
        self.frames[frame_id].print_frame()


def test():
    param = {"length": 100.0,
             "geometry": [[0, 15.0, 3.0], [0, 6.0, 9.0], [0, -6.0, 9.0],
                          [0, -15.0, 3.0], [0, -6.0, -9.0], [0, 6.0, -9.0]
                          ],
             "contour_density": 1,
             "record_animation": False,
             "visualize_line_structure": True,
             "beam_direction": "x",
             "scaling_vector": [1.01, 1.0, 1.02],
             "dofs_input": {
                 "x0": [0.0, 25.0, 50.0, 75.0, 100.0],
                 "y0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "z0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "a0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "b0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "g0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "y": [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.4, 5.0]],
                 "z": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [4.0, 0.0]],
                 "a": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                 "b": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                 "g": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.pi, np.pi/2]],
                 "x": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}}
    s = Structure(param)
    s.apply_transformation_for_structure()
    s.print_structure_element(0)
    s.print_structure_frame(0)


if __name__ == "__main__":
    test()
