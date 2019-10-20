from source.element.CRBeamElement import CRBeamElement

import numpy as np

material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10.}
element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}


def test():
    coords = np.array([[1., 0., 0.], [2., 0., 0.]])
    element = CRBeamElement(material_params, element_params, coords, '3D')

    element._calculate_local_nodal_forces()
    element.TransformationMatrix = element._calculate_initial_local_cs()

    element.Iteration = 1
    element.previous_deformation = element.current_deformation
    element.current_deformation = np.array([0.1, 0.05, 0.04, 0.0, 0.0, 0.0, 0.2, 0.1, 0.03, 0.0, 0.0, 0.0])
    element.TransformationMatrix = element._calculate_transformation_matrix()

    Kc = element._get_local_stiffness_matrix_material()
    Kg = element._get_local_stiffness_matrix_geometry()
    print(Kc)
    print(Kg)


if __name__ == '__main__':
    test()
