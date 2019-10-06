import json
from os.path import join
import numpy as np

from source.postprocess.skin_model.NodeModel import Node


class LineStructure:
    def __init__(self, structure=None):
        """
        initializing line structure with eigenform
        """
        # initializing variables needed by LineStructure
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [], [], []
        self.x0_vec, self.y0_vec, self.z0_vec = [], [], []
        self.s_vec = []  # beam-wise vector
        self.dx_vec, self.dy_vec, self.dz_vec, self.ds_vec = [], [], [], []
        self.theta_x_vec, self.theta_y_vec, self.theta_z_vec = [], [], []
        self.node_positions = {}
        self.dofs = {}
        self.steps = 1

        # initializing beam info with beam structure
        if structure is not None:
            self.beam_direction = structure.beam_direction
            self.beam_length = structure.beam_length
            self.num_of_nodes = structure.num_of_elements
            self.dofs_input = structure.dofs

            self.init()
            self.print_line_structure_info()

    def init(self):
        self.node_positions["x0"] = self.dofs_input["x0"]
        self.node_positions["y0"] = self.dofs_input["y0"]
        self.node_positions["z0"] = self.dofs_input["z0"]
        self.num_of_nodes = len(self.dofs_input["x0"])
        self.steps = len(self.dofs_input["x"])

        for i in range(self.num_of_nodes):
            x = self.node_positions["x0"][i]
            y = self.node_positions["y0"][i]
            z = self.node_positions["z0"][i]
            node = Node(x, y, z)
            self.nodes.append(node)
            self.x0_vec.append(node.x0)
            self.y0_vec.append(node.y0)
            self.z0_vec.append(node.z0)
            self.x_vec.append(node.x0)
            self.y_vec.append(node.y0)
            self.z_vec.append(node.z0)

        # assigning the beam-wise vector
        if self.beam_direction == "x":
            self.s_vec = self.x_vec
        elif self.beam_direction == "y":
            self.s_vec = self.y_vec
        elif self.beam_direction == "z":
            self.s_vec == self.z_vec
        self.update_dofs(0)
        print("Undeformed Nodes added successfully!")

    def update_dofs(self, step):
        dx_vec, dy_vec, dz_vec, theta_x_vec, theta_y_vec, theta_z_vec = [], [], [], [], [], []
        self.dofs["dx"] = self.dofs_input["x"][step]
        self.dofs["dy"] = self.dofs_input["y"][step]
        self.dofs["dz"] = self.dofs_input["z"][step]
        self.dofs["theta_x"] = self.dofs_input["a"][step]
        self.dofs["theta_y"] = self.dofs_input["b"][step]
        self.dofs["theta_z"] = self.dofs_input["g"][step]  # torsion
        for i in range(self.num_of_nodes):
            dx = self.dofs["dx"][i]
            dy = self.dofs["dy"][i]
            dz = self.dofs["dz"][i]
            theta_x = self.dofs["theta_x"][i]
            theta_y = self.dofs["theta_y"][i]
            theta_z = self.dofs["theta_z"][i]
            self.nodes[i].assign_dofs(
                dx, dy, dz, theta_x, theta_y, theta_z)
            dx_vec.append(self.nodes[i].dx)
            dy_vec.append(self.nodes[i].dy)
            dz_vec.append(self.nodes[i].dz)
            theta_x_vec.append(self.nodes[i].theta_x)
            theta_y_vec.append(self.nodes[i].theta_y)
            theta_z_vec.append(self.nodes[i].theta_z)
            self.dx_vec = dx_vec
            self.dy_vec = dy_vec
            self.dz_vec = dz_vec
            self.theta_x_vec = theta_x_vec
            self.theta_y_vec = theta_y_vec
            self.theta_z_vec = theta_z_vec
            # self.nodes[i].print_info()

    def print_line_structure_info(self):
        msg = "=============================================\n"
        msg += "LINE STRUCTURE MODEL INFO \n"
        msg += "NUMBER OF NODES:\t" + str(self.num_of_nodes) + "\n"
        msg += "============================================="
        print(msg)

    def print_nodal_info(self):
        for node in self.nodes:
            node.print_info()

    def apply_transformation_for_line_structure(self):
        for i in range(len(self.nodes)):
            self.nodes[i].apply_transformation()
            self.x_vec[i] = self.nodes[i].x
            self.y_vec[i] = self.nodes[i].y
            self.z_vec[i] = self.nodes[i].z


if __name__ == "__main__":
    ls = LineStructure("trapezoid.json")
