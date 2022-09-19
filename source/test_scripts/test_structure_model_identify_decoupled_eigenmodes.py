# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.test_utils.code_structure import TEST_INPUT_DIRECTORY
import source.auxiliary.global_definitions as GD
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain

# --- STL Imports ---
import unittest.mock as mock
import json


class test_structure_model_identify_decoupled_eigenmodes(TestCase):

    @staticmethod
    def StraightBeamFactory() -> StraightBeam:
        with open(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingStatic.json", "r") as config_file:
            parameters = json.load(config_file)
        return StraightBeam(parameters["model_parameters"])

    # ------------------------------------------------------------------------------------------------------
    # 1. check the selection of considered modes
    # ------------------------------------------------------------------------------------------------------
    def test_considered_modes_all (self):
        beam = self.StraightBeamFactory()

        # expected number of eigenmodes == 3
        beam.dofs_to_keep = np.array([3,4,5])

        beam.domain_size = '3D'

        # sorted list longer than expected result. Code should ignore extra eigenmodes
        beam.eig_freqs_sorted_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        # values from example system longer than 3 to make sure option considered_modes = 'all' ignores the extra value
        beam.decomposed_eigenmodes = {
            'eff_modal_mass': [
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}],
            'rel_contribution':[
                {
                    'a': 0.0,
                    'b': 0.0,
                    'g': 0.01,
                    'x': 0.0,
                    'y': 0.01,
                    'z': 0.0
                },
                {
                    'a': 0.0,
                    'b': 0.01,
                    'g': 0.0,
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.01
                },
                {
                    'a': 0.0,
                    'b': 0.01,
                    'g': 0.0,
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.01
                },
                {
                    'a': 0.0,
                    'b': 0.0,
                    'g': 0.01,
                    'x': 0.0,
                    'y': 0.01,
                    'z': 0.0
                }],
            'rel_participation':[
                {'a': 0.0, 'b': 0.0, 'g': 1.0, 'x': 0.0, 'y': 1.0, 'z': 0.0},
                {'a': 0.0, 'b': 2.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 2.0},
                {'a': 0.0, 'b': 3.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 3.0},
                {'a': 0.0, 'b': 0.0, 'g': 4.0, 'x': 0.0, 'y': 4.0, 'z': 0.0}]
        }

        # StraightBeam::identify_decoupled_eigenmodes calls StraightBeam::decompose_and_quantify_eigenmodes internally,
        # which is not checked by this test case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes'):
            StraightBeam.identify_decoupled_eigenmodes(beam,considered_modes='all')

        self.assertEqual(beam.mode_identification_results['sway_z'],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 1.0}])
        self.assertEqual(beam.mode_identification_results['sway_y'],[{'mode_id': 2, 'eff_modal_mass': 0.0, 'rel_participation': 2.0}, {'mode_id': 3, 'eff_modal_mass': 0.0, 'rel_participation': 3.0}])


    def test_considered_modes_larger_length_dofs_to_keep (self):
        beam = self.StraightBeamFactory()

        # expected number of eigenmodes == 3
        beam.dofs_to_keep = np.array([3,4,5])

        beam.domain_size = '3D'

        # sorted list longer than expected result. Code should ignore extra eigenmodes
        beam.eig_freqs_sorted_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        # values from example system longer than 3 to make sure option considered_modes = 'all' ignores the extra value
        beam.decomposed_eigenmodes = {
            'eff_modal_mass': [
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}],
            'rel_contribution':[
                {
                    'a': 0.0,
                    'b': 0.0,
                    'g': 0.01,
                    'x': 0.0,
                    'y': 0.01,
                    'z': 0.0
                },
                {
                    'a': 0.0,
                    'b': 0.01,
                    'g': 0.0,
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.01
                },
                {
                    'a': 0.0,
                    'b': 0.01,
                    'g': 0.0,
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.01
                },
                {
                    'a': 0.0,
                    'b': 0.0,
                    'g': 0.01,
                    'x': 0.0,
                    'y': 0.01,
                    'z': 0.0
                }],
            'rel_participation':[
                {'a': 0.0, 'b': 0.0, 'g': 1.0, 'x': 0.0, 'y': 1.0, 'z': 0.0},
                {'a': 0.0, 'b': 2.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 2.0},
                {'a': 0.0, 'b': 3.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 3.0},
                {'a': 0.0, 'b': 0.0, 'g': 4.0, 'x': 0.0, 'y': 4.0, 'z': 0.0}]
        }

        # StraightBeam::identify_decoupled_eigenmodes calls StraightBeam::decompose_and_quantify_eigenmodes internally,
        # which is not checked by this test case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes'):
            # considered modes=5 but should reduce it to len(DOFs to keep)=3
            StraightBeam.identify_decoupled_eigenmodes(beam,considered_modes=5)

        self.assertEqual(beam.mode_identification_results['sway_z'],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 1.0}])
        self.assertEqual(beam.mode_identification_results['sway_y'],[{'mode_id': 2, 'eff_modal_mass': 0.0, 'rel_participation': 2.0}, {'mode_id': 3, 'eff_modal_mass': 0.0, 'rel_participation': 3.0}])


    # ------------------------------------------------------------------------------------------------------
    # 2. check label contribution to mode type
    # ------------------------------------------------------------------------------------------------------
    def test_label_contribution_single_category(self):
        beam = self.StraightBeamFactory()

        # expected number of eigenmodes == 1, will keep only the first mode
        beam.dofs_to_keep = np.array([0])

        beam.domain_size = '3D'
        beam.eig_freqs_sorted_indices = [0]



        for category in GD.MODE_CATEGORIZATION[beam.domain_size]:
            # preset with all zeros
            beam.decomposed_eigenmodes = {
                'eff_modal_mass': [
                    {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}],
                'rel_contribution':[
                    {
                        'a': 0.0,
                        'b': 0.0,
                        'g': 0.0,
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0
                    }],
                'rel_participation':[
                    {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}]
            }

            # every label by its own
            for label in GD.MODE_CATEGORIZATION[beam.domain_size][category]:
                beam.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01

                # StraightBeam::identify_decoupled_eigenmodes calls StraightBeam::decompose_and_quantify_eigenmodes internally,
                # which is not checked by this test case => mock it out.
                # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
                with mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes'):
                    # considered modes=5 but should reduce it to len(DOFs to keep)=1
                    StraightBeam.identify_decoupled_eigenmodes(beam,considered_modes=5)

                self.assertEqual(beam.mode_identification_results[category],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}])

                # reset to 0
                beam.decomposed_eigenmodes['rel_contribution'][0][label] = 0.0

            # all labels of category at once
            for label in GD.MODE_CATEGORIZATION[beam.domain_size][category]:
                beam.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01

            # considered modes=5 but should reduce it to len(DOFs to keep)=1
            # StraightBeam::identify_decoupled_eigenmodes calls StraightBeam::decompose_and_quantify_eigenmodes internally,
            # which is not checked by this test case => mock it out.
            # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
            with mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes'):
                StraightBeam.identify_decoupled_eigenmodes(beam,considered_modes=5)
            self.assertEqual(beam.mode_identification_results[category],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}])


    # ------------------------------------------------------------------------------------------------------
    # 3. check label contribution to multiple mode types
    # ------------------------------------------------------------------------------------------------------
    def test_label_contribution_multiple_category(self):
        beam = self.StraightBeamFactory()

        # expected number of eigenmodes == 1, will keep only the first mode
        beam.dofs_to_keep = np.array([0])

        beam.domain_size = '3D'
        beam.eig_freqs_sorted_indices = [0]

        for label in GD.DOF_LABELS[beam.domain_size]:
            # preset with all zeros
            res_categories = []
            beam.decomposed_eigenmodes = {
                'eff_modal_mass': [
                    {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}],
                'rel_contribution':[
                    {
                        'a': 0.0,
                        'b': 0.0,
                        'g': 0.0,
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0
                    }],
                'rel_participation':[
                    {'a': 0.0, 'b': 0.0, 'g': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}]}

            # find categories that include the current label
            for category in GD.MODE_CATEGORIZATION[beam.domain_size]:
                if label in GD.MODE_CATEGORIZATION[beam.domain_size][category]:
                    beam.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01
                    # store that the current category will be part of the assertion later
                    res_categories.append(category)

            # StraightBeam::identify_decoupled_eigenmodes calls StraightBeam::decompose_and_quantify_eigenmodes internally,
            # which is not checked by this test case => mock it out.
            # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
            with mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes'):
                # considered modes=5 but should reduce it to len(DOFs to keep)=1
                StraightBeam.identify_decoupled_eigenmodes(beam,considered_modes=5)

            for category_res in res_categories:
                self.assertEqual(beam.mode_identification_results[category_res],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}])

if __name__ == "__main__":
    TestMain()
