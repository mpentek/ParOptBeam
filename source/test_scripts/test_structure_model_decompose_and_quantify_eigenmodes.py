# --- External Improts ---
import numpy as np
from scipy import linalg

# --- Internal Imports ---
from source.test_utils.code_structure import TEST_INPUT_DIRECTORY
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain

# --- STL Imports ---
import unittest.mock as mock
import json


class test_structure_model_decompose_and_qunatify_eigenmodes(TestCase):

    @staticmethod
    def StraightBeamFactory() -> StraightBeam:
        with open(TEST_INPUT_DIRECTORY / "ProjectParameters3DGenericBuildingStatic.json", "r") as config_file:
            parameters = json.load(config_file)
        return StraightBeam(parameters["model_parameters"])


    def create_mock_self_for_contribution (self):
        beam = self.StraightBeamFactory()
        beam.eig_freqs_sorted_indices = np.array([0])
        beam.dofs_to_keep = np.array([0])
        beam.domain_size = '3D'
        beam.eigen_modes_raw = np.array([
        [-3.53420556e-19], [1.63043899e-19],  [8.50278725e-19],  [-4.77395127e-19],
        [-9.19617850e-19], [-2.30959794e-33], [-4.39066387e-19], [2.25363084e-19],
        [-5.55596213e-20], [9.83695305e-20],  [1.73710162e-34],  [3.33840178e-20],
        [-3.83894809e-20], [1.77468855e-35],  [5.77372089e-03],  [-1.25517271e-20],
        [-1.38232703e-02], [-9.17897402e-03]])
        beam.charact_length = 0.3
        beam.n_elements = 3
        beam.parameters['m'] = [2616.6666666666665, 5233.333333333333, 5233.333333333334, 2616.6666666666674]
        beam.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        beam.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]
        return beam


    def create_mock_self_for_modal_mass (self):
        beam = self.StraightBeamFactory()
        beam.eig_freqs_sorted_indices = np.array([0])
        beam.dofs_to_keep = np.array([0])
        beam.domain_size = '3D'
        beam.eigen_modes_raw = np.ones([18,1])/(np.sqrt(3))
        # --> norm(decomposed_mode) == 1
        beam.charact_length = 1.
        beam.n_elements = 3
        return beam


    def create_mock_self_for_decomposing (self):
        beam = self.StraightBeamFactory()
        beam.eig_freqs_sorted_indices = np.array([0])
        beam.dofs_to_keep = np.array([0])
        beam.domain_size = '3D'
        beam.eigen_modes_raw = np.array([[-1.25758545e-15], [1.82974843e-03], [9.67819983e-17], [-5.39773877e-17],
        [3.26922287e-03],  [3.27133067e-17], [-1.25260404e-16], [-6.40126011e-03],
        [1.13361142e-18], [-2.30083611e-17],  [1.08509315e-02], [-2.88425786e-19],
        [1.84413615e-02],  [1.05665743e-18],  [5.89802514e-19]])
        beam.charact_length = 0.3
        beam.n_elements = 3
        beam.parameters['m'] = [2616.6666666666665, 5233.333333333333, 5233.333333333334, 2616.6666666666674]
        beam.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        beam.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]
        return beam


    # ------------------------------------------------------------------------------------------------------
    # 1. check contribution compared with matlab reference results for every label
    # ------------------------------------------------------------------------------------------------------
    def test_contributions (self):

        beam = self.create_mock_self_for_contribution()
        # StraightBeam::decompose_and_quantify_eigenmodes internally calls StraightBeam::eigenvalue_solve,
        # which is not tested in this case => mock it out.
        #
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch("source.model.structure_model.StraightBeam.eigenvalue_solve"):
            StraightBeam.decompose_and_quantify_eigenmodes(beam)

        reference_map = {label : linalg.norm(beam.eigen_modes_raw[idx:13+idx:6][:,0]) for idx, label in enumerate("xyz")}
        reference_map |= {label: beam.charact_length * linalg.norm(beam.eigen_modes_raw[idx:13+idx:6][:,0]) for idx, label in enumerate("abg",3)}

        self.assertDictAlmostEqual(beam.decomposed_eigenmodes['rel_contribution'][0],
                                   reference_map)


    # ------------------------------------------------------------------------------------------------------
    # 2. check modal mass calculation with unit values for every label
    # ------------------------------------------------------------------------------------------------------
    @TestCase.UniqueReferenceDirectory
    def test_modal_masses (self):

        beam = self.create_mock_self_for_modal_mass()
        beam.parameters['m'] = [1000.0,2000.0,2000.0,1000.0]
        beam.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        beam.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]

        # StraightBeam::decompose_and_quantify_eigenmodes internally calls StraightBeam::eigenvalue_solve,
        # which is not tested in this case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch("source.model.structure_model.StraightBeam.eigenvalue_solve"):
            StraightBeam.decompose_and_quantify_eigenmodes(beam)

        for label in "xyzabg":
            # decomposed_eigenmode == 1
            # symmetric and 3 elements --> storey_mass == 1./2 * total_mass
            # numerator == total_mass
            # denominator == total_mass
            # modal_mass == total_mass
            reference_file_name = "test_modal_masses_" + label + ".csv"
            self.CompareToReferenceFile(
                float(beam.decomposed_eigenmodes['rel_participation'][0][label]),
                self.reference_directory / reference_file_name)


    # ------------------------------------------------------------------------------------------------------
    # 3. check modal mass calculation with reference results from matlab
    # ------------------------------------------------------------------------------------------------------
    @TestCase.UniqueReferenceDirectory
    def test_modal_mass_values_symmetric (self):
        beam = self.create_mock_self_for_modal_mass()
        beam.parameters['m'] = [1000.0,2000.0,2000.0,1000.0]
        beam.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        beam.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]

        # StraightBeam::decompose_and_quantify_eigenmodes internally calls StraightBeam::eigenvalue_solve,
        # which is not tested in this case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch("source.model.structure_model.StraightBeam.eigenvalue_solve"):
            StraightBeam.decompose_and_quantify_eigenmodes(beam)

        for label in "xyzabg":
            reference_file_name = "test_modal_mass_values_symmetric_" + label + ".csv"
            self.CompareToReferenceFile(
                float(beam.decomposed_eigenmodes['rel_participation'][0][label]),
                self.reference_directory / reference_file_name)


    @TestCase.UniqueReferenceDirectory
    def test_modal_mass_values_nonsymmetric (self):
        beam = self.create_mock_self_for_modal_mass()
        beam.parameters['m'] = [1000.0,2000.0,3000.0,4000.0]
        beam.parameters['lz'] = [0.4, 0.4, 0.4, 0.4]
        beam.parameters['ly'] = [0.2, 0.2, 0.2, 0.2]

        # StraightBeam::decompose_and_quantify_eigenmodes internally calls StraightBeam::eigenvalue_solve,
        # which is not tested in this case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch("source.model.structure_model.StraightBeam.eigenvalue_solve"):
            StraightBeam.decompose_and_quantify_eigenmodes(beam)

        #for key, value in beam.decomposed_eigenmodes.items():
        #    print(key, value)

        for label in "xyzabg":
            reference_file_name = "test_modal_mass_value_nonsymmetric_" + label + ".csv"
            self.CompareToReferenceFile(
                float(beam.decomposed_eigenmodes["rel_participation"][0][label]),
                self.reference_directory / reference_file_name)

    # ------------------------------------------------------------------------------------------------------
    # 4. check decomposition for pinned-fixed structure
    # ------------------------------------------------------------------------------------------------------
    #TODO: add test for other BC's
    @TestCase.UniqueReferenceDirectory
    def test_decomposing_eigenmode (self):
        # pinned-fixed 3 Element System
        # after reduction 15 DOF left which are not ordered x,y,z,a,b,g
        # order is: a b g x y z a b g x y z a b g

        beam = self.create_mock_self_for_modal_mass()

        # StraightBeam::decompose_and_quantify_eigenmodes internally calls StraightBeam::eigenvalue_solve,
        # which is not tested in this case => mock it out.
        # Note: this is dangerous because assumptions are made about the test subject's internals. @matekelemen
        with mock.patch("source.model.structure_model.StraightBeam.eigenvalue_solve"):
            StraightBeam.decompose_and_quantify_eigenmodes(beam)

        for label in "xyzabg":
            reference_file_name = "test_decomposing_eigenmode_" + label + ".csv"
            self.CompareToReferenceFile(
                float(beam.decomposed_eigenmodes["rel_contribution"][0][label]),
                self.reference_directory / reference_file_name)


if __name__ == "__main__":
    TestMain()
