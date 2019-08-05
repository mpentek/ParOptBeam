import numpy as np
import json

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.visualize_result_utilities as visualize_result_utilities


class StaticAnalysis(AnalysisType):
    """
    Dervied class for the static analysis of a given structure model        
    """

    def __init__(self, structure_model, name="StaticAnalysis"):

        super().__init__(structure_model, name)

    def solve(self, ext_force):
        print("Solving for ext_force in StaticAnalysis derived class \n")
        self.force = ext_force
        force = self.structure_model.apply_bc_by_reduction(
            self.force, 'row_vector')

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)
        print(k)
        self.static_result = np.linalg.solve(k, force)
        self.static_result = self.structure_model.recuperate_bc_by_extension(
            self.static_result, 'row_vector')
        self.force_action = {"x": np.zeros(0),
                             "y": np.zeros(0),
                             "z": np.zeros(0),
                             "a": np.zeros(0),
                             "b": np.zeros(0),
                             "g": np.zeros(0)}

        #self.force = self.structure_model.recuperate_bc_by_extension(self.force,'row_vector')
        self.resisting_force = self.force - \
            np.dot(self.structure_model.k, self.static_result)
        ixgrid = np.ix_(self.structure_model.bcs_to_keep, [0])
        self.resisting_force[ixgrid] = 0
        self.reaction = {"x": np.zeros(0),
                         "y": np.zeros(0),
                         "z": np.zeros(0),
                         "a": np.zeros(0),
                         "b": np.zeros(0),
                         "g": np.zeros(0)}

    def plot_solve_result(self):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.displacmenet
            self.force
            self.reaction_force
        """

        print("Plotting result in StaticAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.static_result.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.static_result[start:stop +
                                                                               1:step][:, 0]
            self.force_action[label] = self.force[start:stop + 1:step]
            self.reaction[label] = self.resisting_force[start:stop + 1:step][:, 0]

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": [self.force_action["x"],
                              self.force_action["y"],
                              self.force_action["z"]],
                 "reaction": [self.reaction["x"],
                              self.reaction["y"],
                              self.reaction["z"]]}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Static Analysis : "

        visualize_result_utilities.plot_result(plot_title,
                                               geometry,
                                               force,
                                               scaling,
                                               1)

    def write_output_file(self):
        """"
        This function writes out the nodal dofs of the deformed state

        """
        file = open("beam.txt", "w")
        dict = {}
        dict["length"] = max(self.structure_model.nodal_coordinates["x0"])
        dict["num_of_elements"] = len(
            self.structure_model.nodal_coordinates["x0"])
        for key, val in self.structure_model.nodal_coordinates.items():
            dict[key] = val.tolist()

        json_string = json.dumps(dict)

        file.write(json_string)
        file.close()

    def postprocess(self):
        """
        Postprocess something
        """
        print("Postprocessing in StaticAnalysis derived class \n")
        pass
