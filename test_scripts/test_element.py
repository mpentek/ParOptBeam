from source.element.CRBeamElement import CRBeamElement
from source.element.TimoshenkoBeamElement import TimoshenkoBeamElement

import numpy as np


def test_timoshenko_element():
    material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': False}
    element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

    coords = np.array([[1., 0.5, 0.1], [2., 0.2, 0.9]])
    material_params['py'] = 1.0
    material_params['pz'] = 1.0
    material_params['ip'] = 1.0

    element = TimoshenkoBeamElement(material_params, element_params, coords, 0, '3D')
    K = element.get_element_stiffness_matrix()
    print(K)


def test_crbeam_element():
    material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
    element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

    coords = np.array([[1., 0.2, 0.9], [2., 0.2, 0.9]])
    element = CRBeamElement(material_params, element_params, coords, 0, '3D')

    element.Iteration = 1
    element.previous_deformation = element.current_deformation
    element.current_deformation = np.array([0.1, 0.05, 0.04, 0.0, 0.0, 0.0, 0.2, 0.1, 0.03, 0.0, 0.0, 0.0])

    Ke_geo = element._get_element_stiffness_matrix_geometry()
    ke_mat = element._get_element_stiffness_matrix_material()
    K = element.get_element_stiffness_matrix()
    M = element.get_element_mass_matrix()
    np.set_printoptions(precision=1)
    print(M)


if __name__ == '__main__':
    test_crbeam_element()
    # test_timoshenko_element()
