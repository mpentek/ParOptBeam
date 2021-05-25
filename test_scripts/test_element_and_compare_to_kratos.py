from source.element.cr_beam_element import CRBeamElement

import numpy as np

if __name__ == "__main__":
    np.set_printoptions(suppress=False, precision=4, linewidth=100)


    def test_crbeam_element_update_incremental():
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

        try:
            assert (abs(Kd_kratos - Kd) < 10).all()
        except AssertionError:
            msg = "##################################################################################\n"
            msg += "Deformation Stiffness matrix\n"
            msg += "Kd in Kratos:\n" + str(Kd_kratos)
            msg += "\nIt is however:\n" + str(Kd)
            print(msg)

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

        try:
            assert (abs(Ke_mat_kratos - Ke_mat) < 10).all()
        except AssertionError:
            msg = "##################################################################################\n"
            msg += "Material Stiffness matrix\n"
            msg += "Ke_mat in Kratos:\n" + str(Ke_mat_kratos)
            msg += "\nIt is however:\n" + str(Ke_mat)
            print(msg)

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

        try:
            assert (abs(bending_mass_matrix_z - bending_mass_matrix_kratos_z) < 1e-4).all()
            print("Bending mass_matrix z is correct")
        except AssertionError:
            msg = "##################################################################################\n"
            msg += "Bending mass matrix z\n"
            msg += "Me in Kratos:\n" + str(bending_mass_matrix_kratos_z)
            msg += "\nIt is however:\n" + str(bending_mass_matrix_z)
            print(msg)

        bending_mass_matrix_y = element.build_single_mass_matrix(Phiz, CTy, CRy, element.L, -1)

        bending_mass_matrix_kratos_y = np.array([
            [1.13489, -0.137711, -0.663886, -0.0435114],
            [-0.137711, 0.138519, 0.0435114, -0.0410891],
            [-0.663886, 0.0435114, 1.13489, 0.137711],
            [-0.0435114, -0.0410891, 0.137711, 0.138519]
        ])

        try:
            assert (abs(bending_mass_matrix_y - bending_mass_matrix_kratos_y) < 1e-4).all()
            print("Bending mass_matrix y is correct")
        except AssertionError:
            msg = "##################################################################################\n"
            msg += "Bending mass matrix y\n"
            msg += "Me in Kratos:\n" + str(bending_mass_matrix_kratos_y)
            msg += "\nIt is however:\n" + str(bending_mass_matrix_y)
            print(msg)

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

        try:
            assert (abs(Me - Me_kratos) < 1e-2).all()
            print("Mass matrix is correct")
        except AssertionError:
            msg = "##################################################################################\n"
            msg += "Consistent mass matrix\n"
            msg += "Me in Kratos:\n" + str(Me_kratos)
            msg += "\nIt is however:\n" + str(Me)
            print(msg)
