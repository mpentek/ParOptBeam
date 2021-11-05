# --- External Improts ---
import numpy as np
from scipy import linalg

# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain
from source.test_utils.code_structure import TEST_REFERENCE_OUTPUT_DIRECTORY

# --- STL Imports ---
import unittest.mock as mock


class test_structure_model_decompose_and_qunatify_eigenmodes(TestCase):

    def create_mock_self_for_contribution (self):
        mock_self = mock.MagicMock()
        mock_self.eig_freqs_sorted_indices = np.array([0])
        mock_self.dofs_to_keep = np.array([0])
        mock_self.domain_size = '3D'
        mock_self.eigen_modes_raw = np.array([
        [-3.53420556e-19], [1.63043899e-19],  [8.50278725e-19],  [-4.77395127e-19],
        [-9.19617850e-19], [-2.30959794e-33], [-4.39066387e-19], [2.25363084e-19],
        [-5.55596213e-20], [9.83695305e-20],  [1.73710162e-34],  [3.33840178e-20],
        [-3.83894809e-20], [1.77468855e-35],  [5.77372089e-03],  [-1.25517271e-20],
        [-1.38232703e-02], [-9.17897402e-03]])
        mock_self.charact_length = 0.3
        mock_self.n_elements = 3
        mock_self.parameters['m'] = [2616.6666666666665, 5233.333333333333, 5233.333333333334, 2616.6666666666674]
        mock_self.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        mock_self.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]
        mock_self.contribution_matlab_solution = np.array([
            1.69482512506842e-19,	8.34473782500097e-20,
        	0.00173211626700000,	4.87586113078838e-19,
            0.0138232703000000,	0.00917897402000000])
        return mock_self

    def create_mock_self_for_modal_mass (self):
        mock_self = mock.MagicMock()
        mock_self.eig_freqs_sorted_indices = np.array([0])
        mock_self.dofs_to_keep = np.array([0])
        mock_self.domain_size = '3D'
        mock_self.eigen_modes_raw = np.ones([18,1])/(np.sqrt(3))
        # --> norm(decomposed_mode) == 1
        mock_self.charact_length = 1.
        mock_self.n_elements = 3
        mock_self.contribution_matlab_solution = np.array([
            1.69482512506842e-19,	8.34473782500097e-20,
        	0.00173211626700000,	4.87586113078838e-19,
            0.0138232703000000,	0.00917897402000000])
        return mock_self

    def create_mock_self_for_decomposing (self):
        mock_self = mock.MagicMock()
        mock_self.eig_freqs_sorted_indices = np.array([0])
        mock_self.dofs_to_keep = np.array([0])
        mock_self.domain_size = '3D'
        mock_self.eigen_modes_raw = np.array([[-1.25758545e-15], [1.82974843e-03], [9.67819983e-17], [-5.39773877e-17],
        [3.26922287e-03],  [3.27133067e-17], [-1.25260404e-16], [-6.40126011e-03],
        [1.13361142e-18], [-2.30083611e-17],  [1.08509315e-02], [-2.88425786e-19],
        [1.84413615e-02],  [1.05665743e-18],  [5.89802514e-19]])
        mock_self.charact_length = 0.3
        mock_self.n_elements = 3
        mock_self.parameters['m'] = [2616.6666666666665, 5233.333333333333, 5233.333333333334, 2616.6666666666674]
        mock_self.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        mock_self.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]
        return mock_self

    # ------------------------------------------------------------------------------------------------------
    # 1. check contribution compared with matlab reference results for every label
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('builtins.zip')
    @mock.patch('source.model.structure_model.StraightBeam.eigenvalue_solve')
    def test_contributions (self,mock_eigenvalue_solve,mock_zip):

        for [idx,label] in [[0,'a'],[1,'b'],[2,'g'],[3,'x'],[4,'y'],[5,'z']]:
            mock_self = self.create_mock_self_for_contribution()
            mock_zip.return_value = [[idx,label]]
            StraightBeam.decompose_and_quantify_eigenmodes(mock_self)
            if label in ['a', 'b', 'g']:
                self.assertEqual(mock_self.decomposed_eigenmodes['rel_contribution'][0],{label: mock_self.charact_length * linalg.norm(mock_self.eigen_modes_raw[idx:(13+idx):6][:,0])})
            else:
                self.assertEqual(mock_self.decomposed_eigenmodes['rel_contribution'][0],{label: linalg.norm(mock_self.eigen_modes_raw[idx:(13+idx):6][:,0])})
            self.assertIsNone(np.testing.assert_allclose(mock_self.decomposed_eigenmodes['rel_contribution'][0][label], mock_self.contribution_matlab_solution[idx]))
            mock_self.reset_mock(return_value=True)

    # ------------------------------------------------------------------------------------------------------
    # 2. check modal mass calculation with unit values for every label
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('source.model.structure_model.StraightBeam.eigenvalue_solve')
    def test_modal_masses (self,mock_eigenvalue_solve):

        for label in ('a', 'b', 'g', 'x', 'y', 'z'):
            mock_self = self.create_mock_self_for_modal_mass()
            mock_self.parameters['m'] = [1000.0,2000.0,2000.0,1000.0]
            mock_self.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
            mock_self.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]
            # decomposed_eigenmode == 1
            # symmetric and 3 elements --> storey_mass == 1./2 * total_mass
            # numerator == total_mass
            # denominator == total_mass
            # modal_mass == total_mass
            StraightBeam.decompose_and_quantify_eigenmodes(mock_self)

            reference_file_name = "test_modal_masses_" + label + ".csv"
            self.CompareToReferenceFile(
                float(mock_self.decomposed_eigenmodes['rel_participation'][0][label]),
                self.reference_directory / reference_file_name)

            mock_self.reset_mock(return_value=True)

    # ------------------------------------------------------------------------------------------------------
    # 3. check modal mass calculation with reference results from matlab
    # ------------------------------------------------------------------------------------------------------
    @mock.patch('source.model.structure_model.StraightBeam.eigenvalue_solve')
    def test_modal_mass_values_symmetric (self, mock_eigenvalue_solve):
        mock_self = self.create_mock_self_for_modal_mass()
        mock_self.parameters['m'] = [1000.0,2000.0,2000.0,1000.0]
        mock_self.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        mock_self.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]

        StraightBeam.decompose_and_quantify_eigenmodes(mock_self)

        for label in ['x', 'y', 'z', 'a', 'b', 'g']:
            reference_file_name = "test_modal_mass_values_symmetric_" + label + ".csv"
            self.CompareToReferenceFile(
                float(mock_self.decomposed_eigenmodes['rel_participation'][0][label]),
                self.reference_directory / reference_file_name)

    @mock.patch('source.model.structure_model.StraightBeam.eigenvalue_solve')
    def test_modal_mass_values_nonsymmetric (self, mock_eigenvalue_solve):
        mock_self = self.create_mock_self_for_modal_mass()
        mock_self.parameters['m'] = [1000.0,2000.0,3000.0,4000.0]
        mock_self.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        mock_self.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]

        StraightBeam.decompose_and_quantify_eigenmodes(mock_self)
        for key, value in mock_self.decomposed_eigenmodes.items():
            print(key, value)

        for label in ['x', 'y', 'z', 'a', 'b', 'g']:
            reference_file_name = "test_modal_mass_value_nonsymmetric_" + label + ".csv"
            self.CompareToReferenceFile(
                float(mock_self.decomposed_eigenmodes["rel_participation"][0][label]),
                self.reference_directory / reference_file_name)

    # ------------------------------------------------------------------------------------------------------
    # 4. check decomposition for pinned-fixed structure
    # ------------------------------------------------------------------------------------------------------
    #TODO: add test for other BC's
    @mock.patch('source.model.structure_model.StraightBeam.eigenvalue_solve')
    def test_decomposing_eigenmode (self, mock_eigenvalue_solve):
        # pinned-fixed 3 Element System
        # after reduction 15 DOF left which are not ordered x,y,z,a,b,g
        # order is: a b g x y z a b g x y z a b g

        mock_self = self.create_mock_self_for_modal_mass()

        StraightBeam.decompose_and_quantify_eigenmodes(mock_self)
        for label in ['x', 'y', 'z', 'a', 'b', 'g']:
            reference_file_name = "test_decomposing_eigenmode_" + label + ".csv"
            self.CompareToReferenceFile(
                float(mock_self.decomposed_eigenmodes["rel_contribution"][0][label]),
                self.reference_directory / reference_file_name)


    @property
    def reference_directory(self):
        return TEST_REFERENCE_OUTPUT_DIRECTORY / "test_structure_model_decompose_and_quantify_eigenmodes"


if __name__ == "__main__":
    TestMain()