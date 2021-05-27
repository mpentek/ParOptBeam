# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.element.cr_beam_element import CRBeamElement

# --- STL Imports ---
import unittest


class TestElementKratos(unittest.TestCase):

    def test_crbeam_element_update_incremental(self):
        material_params = {'rho': 7850, 'e': 2069000000.0, 'nu': 0.29, 'zeta': 0.05, 'lx_i': 1.2, 'is_nonlinear': True}
        element_params = {'a': 0.0001, 'asy': 0.0, 'asz': 0.0, 'iy': 0.0001, 'iz': 0.0001, 'it': 0.0001}

        coords = np.array([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
        element = CRBeamElement(material_params, element_params, coords, 0, '3D')

        Kd_kratos = np.array([
            [66828.2, 0, 0, 0, 0, 0],
            [0, 172417, 0, 0, 0, 0],
            [0, 0, 172417, 0, 0, 0],
            [0, 0, 0, 172417, 0, 0],
            [0, 0, 0, 0, 517250, 0],
            [0, 0, 0, 0, 0, 517250]
        ])

        Kd = element.Kd_mat

        self.AssertMatrix(Kd, Kd_kratos, delta=10)

        Ke_mat_kratos = np.array([
            [172417, 0, 0, 0, 0, 0, -172417, 0, 0, 0, 0, 0],
            [0, 1.43681e+06, 0, 0, 0, 862083, 0, -1.43681e+06, 0, 0, 0, 862083],
            [0, 0, 1.43681e+06, 0, -862083, 0, 0, 0, -1.43681e+06, 0, -862083, 0],
            [0, 0, 0, 66828.2, 0, 0, 0, 0, 0, -66828.2, 0, 0],
            [0, 0, -862083, 0, 689667, 0, 0, 0, 862083, 0, 344833, 0],
            [0, 862083, 0, 0, 0, 689667, 0, -862083, 0, 0, 0, 344833],
            [-172417, 0, 0, 0, 0, 0, 172417, 0, 0, 0, 0, 0],
            [0, -1.43681e+06, 0, 0, 0, -862083, 0, 1.43681e+06, 0, 0, 0, -862083],
            [0, 0, -1.43681e+06, 0, 862083, 0, 0, 0, 1.43681e+06, 0, 862083, 0],
            [0, 0, 0, -66828.2, 0, 0, 0, 0, 0, 66828.2, 0, 0],
            [0, 0, -862083, 0, 344833, 0, 0, 0, 862083, 0, 689667, 0],
            [0, 862083, 0, 0, 0, 344833, 0, -862083, 0, 0, 0, 689667]
        ])

        Ke_mat = element.Ke_mat

        self.AssertMatrix(Ke_mat, Ke_mat_kratos, delta=10)

        Phiz = 0.0
        Phiy = 0.0

        CTy = (element.rho * element.A * element.L) / ((1 + Phiy) * (1 + Phiy))
        CTz = (element.rho * element.A * element.L) / ((1 + Phiz) * (1 + Phiz))

        CRy = (element.rho * element.Iy) / ((1 + Phiy) * (1 + Phiy) * element.L)
        CRz = (element.rho * element.Iz) / ((1 + Phiz) * (1 + Phiz) * element.L)

        bending_mass_matrix_z = element.build_single_mass_matrix(Phiz, CTz, CRz, element.L, +1)

        bending_mass_matrix_kratos_z = np.array([
            [1.13489, 0.137711, -0.663886, 0.0435114],
            [0.137711, 0.138519, -0.0435114, -0.0410891],
            [-0.663886, -0.0435114, 1.13489, -0.137711],
            [0.0435114, -0.0410891, -0.137711, 0.138519]
        ])

        self.AssertMatrix(bending_mass_matrix_z, bending_mass_matrix_kratos_z, delta=10)

        bending_mass_matrix_y = element.build_single_mass_matrix(Phiz, CTy, CRy, element.L, -1)

        bending_mass_matrix_kratos_y = np.array([
            [1.13489, -0.137711, -0.663886, -0.0435114],
            [-0.137711, 0.138519, 0.0435114, -0.0410891],
            [-0.663886, 0.0435114, 1.13489, 0.137711],
            [-0.0435114, -0.0410891, 0.137711, 0.138519]
        ])

        self.AssertMatrix(bending_mass_matrix_y, bending_mass_matrix_kratos_y, delta=10)

        Me = element._get_consistent_mass_matrix()

        Me_kratos = np.array([
            [0.314, 0, 0, 0, 0, 0, 0.157, 0, 0, 0, 0, 0],
            [0, 1.13489, 0, 0, 0, 0.137711, 0, -0.663886, 0, 0, 0, 0.0435114],
            [0, 0, 1.13489, 0, -0.137711, 0, 0, 0, -0.663886, 0, -0.0435114, 0],
            [0, 0, 0, 0.628, 0, 0, 0, 0, 0, 0.314, 0, 0],
            [0, 0, -0.137711, 0, 0.138519, 0, 0, 0, 0.0435114, 0, -0.0410891, 0],
            [0, 0.137711, 0, 0, 0, 0.138519, 0, -0.0435114, 0, 0, 0, -0.0410891],
            [0.157, 0, 0, 0, 0, 0, 0.314, 0, 0, 0, 0, 0],
            [0, -0.663886, 0, 0, 0, -0.0435114, 0, 1.13489, 0, 0, 0, -0.137711],
            [0, 0, -0.663886, 0, 0.0435114, 0, 0, 0, 1.13489, 0, 0.137711, 0],
            [0, 0, 0, 0.314, 0, 0, 0, 0, 0, 0.628, 0, 0],
            [0, 0, -0.0435114, 0, -0.0410891, 0, 0, 0, 0.137711, 0, 0.138519, 0],
            [0, 0.0435114, 0, 0, 0, -0.0410891, 0, -0.137711, 0, 0, 0, 0.138519]
        ])

        self.AssertMatrix(Me, Me_kratos, delta=10)


    def AssertArray(self, array, reference, **kwargs):
        for item, item_reference in zip(array, reference):
            self.assertAlmostEqual(item, item_reference, **kwargs)


    def AssertMatrix(self, matrix: np.ndarray, reference: np.ndarray, **kwargs):
        for row, row_reference in zip(matrix, reference):
            self.AssertArray(row, row_reference, **kwargs)


if __name__ == "__main__":
    unittest.main()