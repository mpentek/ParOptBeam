# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.element.cr_beam_element import CRBeamElement
from source.element.timoshenko_beam_element import TimoshenkoBeamElement
from source.test_utils.code_structure import TEST_REFERENCE_OUTPUT_DIRECTORY
from source.test_utils.test_case import TestCase, TestMain


class TestElement(TestCase):

    def test_timoshenko_element(self):
        material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': False}
        element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

        coords = np.array([[1., 0.5, 0.1], [2., 0.2, 0.9]])
        material_params['py'] = 1.0
        material_params['pz'] = 1.0
        material_params['ip'] = 1.0

        element = TimoshenkoBeamElement(material_params, element_params, coords, 0, '3D')
        self.CompareToReferenceFile(
            element.get_element_stiffness_matrix(),
            self.reference_directory / "k.csv",
            delta=self.tolerance)


    def test_crbeam_element_update_incremental(self):
        material_params = {'rho': 1000.0, 'e': 1.e6, 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
        element_params = {'a': 1., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

        coords = np.array([[1., 0.0, 0.0], [2.0, 0.0, 0.0]])
        element = CRBeamElement(material_params, element_params, coords, 0, '3D')

        ke_mat_1 = element._get_element_stiffness_matrix_material()

        dp = [0.2, 0.6, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        element.update_incremental(dp)
        new_coords = element._get_current_nodal_position()
        new_coords_sol = [1.2, 0.6, 0.0, 2.0, 0.0, 0.0]

        self.CompareToReferenceFile(
            element._get_current_nodal_position(),
            self.reference_directory / "nodal_position.csv",
            delta=self.tolerance)

        self.assertAlmostEqual(
            element._calculate_current_length(),
            np.sqrt(0.8 ** 2 + 0.6 ** 2),
            delta=self.tolerance)

        self.CompareToReferenceFile(
            element._calculate_transformation_s(),
            self.reference_directory / "transformation_s.csv",
            delta=self.tolerance)

        self.CompareToReferenceFile(
            element.LocalRotationMatrix,
            self.reference_directory / "rotation_matrix.csv",
            delta=self.tolerance)

        self.CompareToReferenceFile(
            element.v,
            self.reference_directory / "dv.csv",
            delta=self.tolerance)

        ke_mat_2 = element._get_element_stiffness_matrix_material()

        self.assertMatrixAlmostEqual(
            ke_mat_1,
            ke_mat_2,
            delta=self.tolerance)

        self.CompareToReferenceFile(
            element.get_element_stiffness_matrix(),
            self.reference_directory / "k.csv",
            delta=self.tolerance)

        self.CompareToReferenceFile(
            element.nodal_force_global,
            self.reference_directory / "q.csv",
            delta=self.tolerance)


    def test_crbeam_element_update_total(self):
        material_params = {'rho': 1000.0, 'e': 1.e6, 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
        element_params = {'a': 1., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

        coords = np.array([[100., 0.0, 0.0], [200.0, 0.0, 0.0]])
        element = CRBeamElement(material_params, element_params, coords, 0, '3D')

        dp_v = [0.1, 0.4, 0.1, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        element.update_total(dp_v)
        K = element.get_element_stiffness_matrix()
        f_test = np.dot(K, dp_v)
        q = element.nodal_force_global

        # Failure
        #self.assertArrayAlmostEqual(f_test, q, delta=self.tolerance)


    @property
    def tolerance(self):
        return 1e-6


    @property
    def reference_directory(self):
        return TEST_REFERENCE_OUTPUT_DIRECTORY / "test_element"


if __name__ == "__main__":
    TestMain()