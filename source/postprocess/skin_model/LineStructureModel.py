import numpy as np
from source.postprocess.skin_model.NodeModel import Node
from joblib import Parallel, delayed
import multiprocessing

NUM_OF_CORES = multiprocessing.cpu_count()


class LineStructure:
    def __init__(self, params):
        """
        initializing line structure with dofs
        """
        # initializing beam info with param
        self.params = params
        self.beam_length = params["length"]
        self.num_of_nodes = len(params["dofs_input"]["x0"])
        self.dofs_input = params["dofs_input"]

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
        self.steps = len(self.dofs_input["x"][0])

        for i in range(self.num_of_nodes):
            position = [self.dofs_input["x0"][i],
                        self.dofs_input["y0"][i],
                        self.dofs_input["z0"][i]]
            self.nodes[i] = Node(position)
            self.undeformed[0][i] = self.nodes[i].undeformed[0]
            self.undeformed[1][i] = self.nodes[i].undeformed[1]
            self.undeformed[2][i] = self.nodes[i].undeformed[2]
            self.deformed[0][i] = self.nodes[i].deformed[2]
            self.deformed[1][i] = self.nodes[i].deformed[2]
            self.deformed[2][i] = self.nodes[i].deformed[2]

        self.update_dofs(0)

        print("Undeformed Nodes added successfully!")

    def update_dofs(self, step):

        x = np.asarray(self.dofs_input["x"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]
        y = np.asarray(self.dofs_input["y"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]
        z = np.asarray(self.dofs_input["z"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]

        a = np.asarray(self.dofs_input["a"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]
        b = np.asarray(self.dofs_input["b"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]
        g = np.asarray(self.dofs_input["g"]).reshape(self.steps * self.num_of_nodes)[step::self.steps]

        self.displacement = np.array([x, y, z])
        self.angular_displacement = np.array([a, b, g])

        displacement = self.displacement.transpose().reshape(self.num_of_nodes, 3)
        angular_displacement = self.angular_displacement.transpose().reshape(self.num_of_nodes, 3)

        def assign_nodal_dof(i):
            self.nodes[i].assign_dofs(displacement[i], angular_displacement[i])
            return self.nodes[i]

        self.nodes = Parallel(n_jobs=NUM_OF_CORES)(delayed(assign_nodal_dof)(i) for i in range(self.num_of_nodes))

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

        def apply_nodal_transformation(i):
            self.nodes[i].apply_transformation()
            self.deformed[0][i] = self.nodes[i].deformed[0]
            self.deformed[1][i] = self.nodes[i].deformed[1]
            self.deformed[2][i] = self.nodes[i].deformed[2]
            merged = [self.nodes[i], self.deformed[0][i], self.deformed[1][i], self.deformed[2][i]]
            return merged

        merged_solution = Parallel(n_jobs=NUM_OF_CORES)(delayed(apply_nodal_transformation)(i)
                                                        for i in range(self.num_of_nodes))

        merged_solution = np.asarray(merged_solution).transpose()
        self.nodes = merged_solution[0]
        self.deformed[0] = merged_solution[1]
        self.deformed[1] = merged_solution[2]
        self.deformed[2] = merged_solution[3]


def test():
    param = {"length": 100.0,
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
                 "g": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.pi, np.pi / 2]],
                 "x": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}}
    ls = LineStructure(param)
    ls.apply_transformation_for_line_structure()
    solution = np.array([-100, 0.4, 4.0])
    err = solution - ls.nodes[4].deformed
    print(ls.nodes[4].deformed)
    try:
        assert all(err < 1e-12), "Transformation wrong"
        print("passed test")
    except AssertionError:
        print("failed test")


if __name__ == "__main__":
    test()
