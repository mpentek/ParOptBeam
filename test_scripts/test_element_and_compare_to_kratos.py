from source.element.CRBeamElement import CRBeamElement

import numpy as np


TOL = 10


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
        assert (abs(Kd_kratos - Kd) < TOL).all()
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
        assert (abs(Ke_mat_kratos - Ke_mat) < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Material Stiffness matrix\n"
        msg += "Kd in Kratos:\n" + str(Ke_mat_kratos)
        msg += "\nIt is however:\n" + str(Ke_mat)
        print(msg)
