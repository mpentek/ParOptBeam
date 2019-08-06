import numpy as np
import json
import os

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.postprocess.visualize_result_utilities as visualize_result_utilities
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults


class StaticAnalysis(AnalysisType):
    """
    Dervied class for the static analysis of a given structure model        
    """

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "type": "static_analysis",
        "settings": {},
        "input": {},
        "output": {}}

    def __init__(self, structure_model, parameters):

        # validating and assign model parameters
        validate_and_assign_defaults(
            StaticAnalysis.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        super().__init__(structure_model, self.parameters["type"])

        selected_time_step = 15000

        # load parameters
        '''
        FOR NOW ONLY AVAILABLE for
        1 elements - 2 nodes
        2 elements - 3 nodes
        3 elements - 4 nodes
        6 elements - 7 nodes
        12 elements - 13 nodes
        24 elements - 25 nodes
        '''
        possible_n_el_cases = [1, 2, 3, 6, 12, 24]
        if structure_model.parameters['n_el'] not in possible_n_el_cases:
            err_msg = "The number of element input \"" + \
                str(structure_model.parameters['n_el'])
            err_msg += "\" is not allowed for Dynamic Analysis \n"
            err_msg += "Choose one of: "
            err_msg += ', '.join([str(x) for x in possible_n_el_cases])
            raise Exception(err_msg)
        # TODO include some specifiers in the parameters, do not hard code
        self.force = np.load(os.path.join(*['input', 'force', 'force_dynamic' + '_turb' + str(
            structure_model.parameters['n_el']+1) + '.npy']))[:, selected_time_step]

    def solve(self):
        print("Solving for ext_force in StaticAnalysis derived class \n")
        # self.force = ext_force
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
        ixgrid = np.ix_(self.structure_model.dofs_to_keep, [0])
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

        for plot_result in self.parameters['output']['plot']:
            if plot_result == 'deformation':
                self.plot_solve_result()
            if plot_result == 'forces':
                pass

        for write_result in self.parameters['output']['write']:
            pass

        pass
