# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.element.cr_beam_element import CRBeamElement
from source.test_utils.test_case import TestCase, TestMain


class TestElementKratos(TestCase):

    @TestCase.UniqueReferenceDirectory
    def test_crbeam_element_update_incremental(self):
        material_params = {'rho': 7850, 'e': 2069000000.0, 'nu': 0.29, 'zeta': 0.05, 'lx_i': 1.2, 'is_nonlinear': True}
        element_params = {'a': 0.0001, 'asy': 0.0, 'asz': 0.0, 'iy': 0.0001, 'iz': 0.0001, 'it': 0.0001}

        coords = np.array([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
        element = CRBeamElement(material_params, element_params, coords, 0, '3D')

        self.CompareToReferenceFile(
            element.Kd_mat,
            self.reference_directory / "crbeam_update_incremental_kd.csv",
            delta=10)

        self.CompareToReferenceFile(
            element.Ke_mat,
            self.reference_directory / "crbeam_update_incremental_ke.csv",
            delta=10)

        Phiz = 0.0
        Phiy = 0.0

        CTy = (element.rho * element.A * element.L) / ((1 + Phiy) * (1 + Phiy))
        CTz = (element.rho * element.A * element.L) / ((1 + Phiz) * (1 + Phiz))

        CRy = (element.rho * element.Iy) / ((1 + Phiy) * (1 + Phiy) * element.L)
        CRz = (element.rho * element.Iz) / ((1 + Phiz) * (1 + Phiz) * element.L)

        bending_mass_matrix_z = element.build_single_mass_matrix(Phiz, CTz, CRz, element.L, +1)


        self.CompareToReferenceFile(
            bending_mass_matrix_z,
            self.reference_directory / "crbeam_update_incremental_bending_mass_z.csv",
            delta=10)

        bending_mass_matrix_y = element.build_single_mass_matrix(Phiz, CTy, CRy, element.L, -1)

        self.CompareToReferenceFile(
            bending_mass_matrix_y,
            self.reference_directory / "crbeam_update_incremental_bending_mass_y.csv",
            delta=10)

        self.CompareToReferenceFile(
            element._get_consistent_mass_matrix(),
            self.reference_directory / "crbeam_update_incremental_me.csv",
            delta=10)


if __name__ == "__main__":
    TestMain()