import unittest
import unittest.mock as mock
import scipy
import numpy as np
from source.model.structure_model import StraightBeam
import source.auxiliary.global_definitions as GD


class test_structure_model_identify_decoupled_eigenmodes(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------
    # 1. check the selection of considered modes
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes')
    def test_considered_modes_all (self,mock_decompose_and_quantify_eigenmodes):
        mock_self = mock.MagicMock()

        # expected number of eigenmodes == 3
        mock_self.dofs_to_keep = np.array([3,4,5])

        mock_self.domain_size = '3D'

        # sorted list longer than expected result. Code should ignore extra eigenmodes
        mock_self.eig_freqs_sorted_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        # values from example system longer than 3 to make sure option considered_modes = 'all' ignores the extra value
        mock_self.decomposed_eigenmodes = {
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
        StraightBeam.identify_decoupled_eigenmodes(mock_self,considered_modes='all')

        self.assertEqual(mock_self.mode_identification_results['sway_z'],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 1.0}])
        self.assertEqual(mock_self.mode_identification_results['sway_y'],[{'mode_id': 2, 'eff_modal_mass': 0.0, 'rel_participation': 2.0}, {'mode_id': 3, 'eff_modal_mass': 0.0, 'rel_participation': 3.0}])

    @mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes')
    def test_considered_modes_larger_length_dofs_to_keep (self,mock_decompose_and_quantify_eigenmodes):
        mock_self = mock.MagicMock()

        # expected number of eigenmodes == 3
        mock_self.dofs_to_keep = np.array([3,4,5])

        mock_self.domain_size = '3D'

        # sorted list longer than expected result. Code should ignore extra eigenmodes
        mock_self.eig_freqs_sorted_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        # values from example system longer than 3 to make sure option considered_modes = 'all' ignores the extra value
        mock_self.decomposed_eigenmodes = {
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

        # considered modes=5 but should reduce it to len(DOFs to keep)=3
        StraightBeam.identify_decoupled_eigenmodes(mock_self,considered_modes=5)

        self.assertEqual(mock_self.mode_identification_results['sway_z'],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 1.0}])
        self.assertEqual(mock_self.mode_identification_results['sway_y'],[{'mode_id': 2, 'eff_modal_mass': 0.0, 'rel_participation': 2.0}, {'mode_id': 3, 'eff_modal_mass': 0.0, 'rel_participation': 3.0}])
    
    
    # ------------------------------------------------------------------------------------------------------
    # 2. check label contribution to mode type
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes')
    def test_label_contribution_single_category(self,mock_decompose_and_quantify_eigenmodes):
        mock_self = mock.MagicMock()

        # expected number of eigenmodes == 1, will keep only the first mode
        mock_self.dofs_to_keep = np.array([0])

        mock_self.domain_size = '3D'
        mock_self.eig_freqs_sorted_indices = [0]



        for category in GD.MODE_CATEGORIZATION[mock_self.domain_size]:
            # preset with all zeros
            mock_self.decomposed_eigenmodes = {
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
            for label in GD.MODE_CATEGORIZATION[mock_self.domain_size][category]:
                mock_self.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01

                # considered modes=5 but should reduce it to len(DOFs to keep)=1
                StraightBeam.identify_decoupled_eigenmodes(mock_self,considered_modes=5)

                self.assertEqual(mock_self.mode_identification_results[category],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}])
                
                # reset to 0
                mock_self.decomposed_eigenmodes['rel_contribution'][0][label] = 0.0
            
            # all labels of category at once
            for label in GD.MODE_CATEGORIZATION[mock_self.domain_size][category]:
                mock_self.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01

            # considered modes=5 but should reduce it to len(DOFs to keep)=1
            StraightBeam.identify_decoupled_eigenmodes(mock_self,considered_modes=5)
            self.assertEqual(mock_self.mode_identification_results[category],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}]) 


    # ------------------------------------------------------------------------------------------------------
    # 3. check label contribution to multiple mode types
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('source.model.structure_model.StraightBeam.decompose_and_quantify_eigenmodes')
    def test_label_contribution_multiple_category(self,mock_decompose_and_quantify_eigenmodes):
        mock_self = mock.MagicMock()

        # expected number of eigenmodes == 1, will keep only the first mode
        mock_self.dofs_to_keep = np.array([0])

        mock_self.domain_size = '3D'
        mock_self.eig_freqs_sorted_indices = [0]

        for label in GD.DOF_LABELS[mock_self.domain_size]:
            # preset with all zeros
            res_categories = []
            mock_self.decomposed_eigenmodes = {
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
            for category in GD.MODE_CATEGORIZATION[mock_self.domain_size]:
                if label in GD.MODE_CATEGORIZATION[mock_self.domain_size][category]:
                    mock_self.decomposed_eigenmodes['rel_contribution'][0][label] = 0.01
                    # store that the current category will be part of the assertion later
                    res_categories.append(category)

            # considered modes=5 but should reduce it to len(DOFs to keep)=1
            StraightBeam.identify_decoupled_eigenmodes(mock_self,considered_modes=5)
            for category_res in res_categories:
                self.assertEqual(mock_self.mode_identification_results[category_res],[{'mode_id': 1, 'eff_modal_mass': 0.0, 'rel_participation': 0.0}])
 
if __name__ == "__main__":
    unittest.main()     