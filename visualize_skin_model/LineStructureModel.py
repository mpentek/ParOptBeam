import json
from visualize_skin_model.NodeModel import Node


class LineStructure:
    def __init__(self, structure_file=None):
        """
        initializing line structure with eigenform
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [], [], []
        self.dx_vec, self.dy_vec, self.dz_vec = [], [], []
        self.theta_x_vec, self.theta_y_vec, self.theta_z_vec = [], [], []
        if structure_file is not None:
            with open(structure_file) as json_file:
                data = json.load(json_file)
                self.num_of_dofs_per_node = data["num_of_dofs_per_node"]
                self.node_positions = data["node_positions"]
                self.dofs = data["dofs"]
                self.num_of_nodes = int(len(self.dofs)/self.num_of_dofs_per_node)
                self.structure_height = data["height"]
            self.print_line_structure_info()
            self.init_nodes()
            self.init_dofs()

    def init_nodes(self):
        for i in range(self.num_of_nodes):
            x = self.node_positions[int(i*3  )]
            y = self.node_positions[int(i*3+1)]
            z = self.node_positions[int(i*3+2)]
            node = Node(x, y, z)
            self.nodes.append(node)
            self.x_vec.append(node.x0)
            self.y_vec.append(node.y0)
            self.z_vec.append(node.z0)

    def init_dofs(self):
        for i in range(self.num_of_nodes):
            dx = self.dofs[int(i*self.num_of_dofs_per_node  )]
            dy = self.dofs[int(i*self.num_of_dofs_per_node+1)]
            dz = self.dofs[int(i*self.num_of_dofs_per_node+2)]
            theta_x = self.dofs[int(i*self.num_of_dofs_per_node+3)]
            theta_y = self.dofs[int(i*self.num_of_dofs_per_node+4)]
            theta_z = self.dofs[int(i*self.num_of_dofs_per_node+5)]
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

    def print_nodal_infos(self):
        for node in self.nodes:
            node.print_info()

    def apply_transformation_for_line_structure(self):
        for i in range(len(self.nodes)):
            self.nodes[i].apply_transformation()
            self.x_vec[i] = self.nodes[i].x
            self.y_vec[i] = self.nodes[i].y
            self.z_vec[i] = self.nodes[i].z
