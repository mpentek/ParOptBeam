import json
import numpy as np

try:
    from visualize_skin_model.NodeModel import Node
except ModuleNotFoundError:
    from NodeModel import Node


class LineStructure:
    def __init__(self, structure_file=None):
        """
        initializing line structure with eigenform
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [], [], []
        self.s_vec = []  # beam-wise vector
        self.dx_vec, self.dy_vec, self.dz_vec, self.ds_vec = [], [], [], []
        self.theta_x_vec, self.theta_y_vec, self.theta_z_vec = [], [], []
        self.beam_length = 0
        self.num_of_nodes = 0
        self.node_positions = {}
        self.dofs = {}
        self.beam_direction = None

        if structure_file is not None:
            with open(structure_file) as json_file:
                data = json.load(json_file)
                self.num_of_dofs_per_node = data["num_of_dofs_per_node"]
                self.dof_file = data["dofs_file_name"]
                self.beam_direction = data["beam_direction"]

            self.init_nodes()
            self.init_dofs()
            self.print_line_structure_info()

    def init_nodes(self):

        with open(self.dof_file) as f:
            f = json.load(f)
            # wrapping beam direction from x to z (mdof_solver)
            self.beam_length = f["length"]
            self.num_of_nodes = f["num_of_elements"]
            self.node_positions["x0"] = f["x0"]
            self.node_positions["y0"] = f["y0"]
            self.node_positions["z0"] = f["z0"]

            self.num_of_nodes = len(f["x0"])

        for i in range(self.num_of_nodes):
            x = self.node_positions["x0"][i]
            y = self.node_positions["y0"][i]
            z = self.node_positions["z0"][i]
            node = Node(x, y, z)
            self.nodes.append(node)
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
        print("Undeformed Nodes added successfully!")

    def init_dofs(self):
        with open(self.dof_file) as f:
            f = json.load(f)
            # TODO check the dof in beam direction
            self.dofs["dx"] = np.zeros(self.num_of_nodes)
            self.dofs["dy"] = f["y"]
            self.dofs["dz"] = f["z"]
            self.dofs["theta_x"] = f["a"]
            self.dofs["theta_y"] = f["b"]
            self.dofs["theta_z"] = f["g"]  # torsion

            for i in range(self.num_of_nodes):
                dx = self.dofs["dx"][i]
                dy = self.dofs["dy"][i]
                dz = self.dofs["dz"][i]
                theta_x = self.dofs["theta_x"][i]
                theta_y = self.dofs["theta_y"][i]
                theta_z = self.dofs["theta_z"][i]
                self.nodes[i].assign_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
                self.dx_vec.append(self.nodes[i].dx)
                self.dy_vec.append(self.nodes[i].dy)
                self.dz_vec.append(self.nodes[i].dz)
                self.theta_x_vec.append(self.nodes[i].theta_x)
                self.theta_y_vec.append(self.nodes[i].theta_y)
                self.theta_z_vec.append(self.nodes[i].theta_z)
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