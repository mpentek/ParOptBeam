from os.path import join
import numpy as np
from source.postprocess.skin_model.NodeModel import Node


class LineStructure:
    def __init__(self, params):
        """
        initializing line structure with dofs
        """
        # Setting default values
        self.beam_length = 1
        self.num_of_nodes = 1

        # initializing beam info with param
        self.beam_length = params["length"]
        self.num_of_nodes = params["num_of_elements"]
        self.dofs_input = params

        # initializing variables needed by LineStructure
        self.nodes = np.empty(self.num_of_nodes, dtype=Node)
        self.undeformed = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.deformed = np.ndarray((3, self.num_of_nodes), dtype=float)

        self.displacement = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.angular_displacement = np.ndarray((3, self.num_of_nodes), dtype=float)
        self.steps = 1

        self.init()
        self.print_line_structure_info()

    def init(self):
        self.steps = len(self.dofs_input["x"])

        for i in range(self.num_of_nodes):
            position = [self.dofs_input["x0"][i],
                        self.dofs_input["y0"][i],
                        self.dofs_input["z0"][i]]
            self.nodes[i] = Node(position)
            self.nodes[i].print_info()
            self.undeformed[0][i] = self.nodes[i].undeformed[0]
            self.undeformed[1][i] = self.nodes[i].undeformed[1]
            self.undeformed[2][i] = self.nodes[i].undeformed[2]

        self.deformed = self.undeformed
        self.update_dofs(0)
        print("Undeformed Nodes added successfully!")

    def update_dofs(self, step):
        displacement = np.array([self.dofs_input["x"][step],
                                 self.dofs_input["y"][step],
                                 self.dofs_input["z"][step]])
        angular_displacement = np.array([self.dofs_input["a"][step],
                                         self.dofs_input["b"][step],
                                         self.dofs_input["g"][step]])

        displacement = displacement.transpose().reshape(self.num_of_nodes, 3)
        angular_displacement = angular_displacement.transpose().reshape(self.num_of_nodes, 3)
        for i in range(self.num_of_nodes):
            self.nodes[i].assign_dofs(displacement[i], angular_displacement[i])

    def print_line_structure_info(self):
        msg = "=============================================\n"
        msg += "LINE STRUCTURE MODEL INFO \n"
        msg += "NUMBER OF NODES:\t" + str(self.num_of_nodes) + "\n"
        msg += "=============================================\n"
        print(msg)

    def print_nodal_info(self):
        for node in self.nodes:
            node.print_info()

    def apply_transformation_for_line_structure(self):
        for i in range(self.num_of_nodes):
            self.nodes[i].apply_transformation()
            self.deformed[0][i] = self.nodes[i].deformed[0]
            self.deformed[1][i] = self.nodes[i].deformed[1]
            self.deformed[2][i] = self.nodes[i].deformed[2]


def test():
    param = {"length": 100.0, "num_of_elements": 5,
             "x0": [0.0, 25.0, 50.0, 75.0, 100.0],
             "y0": [0.0, 0.0, 0.0, 0.0, 0.0],
             "z0": [0.0, 0.0, 0.0, 0.0, 0.0],
             "a0": [0.0, 0.0, 0.0, 0.0, 0.0],
             "b0": [0.0, 0.0, 0.0, 0.0, 0.0],
             "g0": [0.0, 0.0, 0.0, 0.0, 0.0],
             "y": [[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 1.1, 2.2, 3.3, 4.4]],
             "z": [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]],
             "a": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
             "b": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
             "g": [[0.0, 0.0, 0.0, 0.0, np.pi], [0.0, 0.0, 0.0, 0.0, 3.14]],
             "x": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]}
    ls = LineStructure(param)
    ls.apply_transformation_for_line_structure()
    ls.print_nodal_info()
    solution = np.array([-100, 0.4, 4.0])

    try:
        assert all(ls.nodes[4].deformed == solution), "Transformation wrong"
        print("passed test")
    except AssertionError:
        print("failed test")


if __name__ == "__main__":
    test()
