from source.element.CRBeamElement import CRBeamElement

import numpy as np

material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10.}
element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}



def test_crbeam_element():
    coords = np.array([[1., 0.5, 0.1], [2., 0.2, 0.9]])
    element = CRBeamElement(material_params, element_params, coords, '3D')

    LHS = element.get_element_stiffness_matrix()
    print(LHS)

    element.Iteration = 1
    element.previous_deformation = element.current_deformation
    element.current_deformation = np.array([0.1, 0.05, 0.04, 0.0, 0.0, 0.0, 0.2, 0.1, 0.03, 0.0, 0.0, 0.0])

    LHS = element.get_element_stiffness_matrix()
    print(LHS)


if __name__ == '__main__':
    # test_crbeam_element()
