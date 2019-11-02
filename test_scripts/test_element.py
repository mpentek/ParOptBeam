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


TOL = 1e-12


def test_crbeam_element():
    material_params = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
    element_params = {'a': 5., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

    coords = np.array([[1., 0.0, 0.0], [2.0, 0.0, 0.0]])
    element = CRBeamElement(material_params, element_params, coords, 0, '3D')

    ke_mat_1 = element._get_element_stiffness_matrix_material()
    K = element.get_element_stiffness_matrix()

    dp = [0.2, 0.1, 0.0, 0.2, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    element.assign_new_deformation(dp)
    new_coords = element._get_current_nodal_position()
    new_coords_sol = [1.2, 0.1, 0.0, 2.0, 0.0, 0.0]

    try:
        assert (new_coords - new_coords_sol < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in coordinate update\n"
        msg += "New coordinate is suppose to be:\n" + str(new_coords_sol)
        msg += "\nIt is however:\n" + str(new_coords)
        print(msg)

    l = element._calculate_current_length()
    l_sol = np.sqrt(0.8**2 + 0.1**2)

    try:
        assert l - l_sol < TOL
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in current length calculation\n"
        msg += "Current length is suppose to be:\n" + str(l_sol)
        msg += "\nIt is however:\n" + str(l)
        print(msg)

    S = element._calculate_transformation_s()

    S_sol = np.array([
        [0., 0., 0., -1, 0., 0.],
        [0., 0., 0., 0., 0., 2 / l],
        [0., 0., 0., 0., -2 / l, 0.],
        [-1., 0., 0., 0., 0., 0.],
        [0., -1., 0., 0., 1., 0.],
        [0., 0., -1., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., -2 / l],
        [0., 0., 0., 0., 2 / l, 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 1.],
    ])

    try:
        assert (S - S_sol < TOL).all()
    except AssertionError:
        print("Mistake in local transformation matrix S calculation")

    element.update_internal_force()

    r0 = element.Quaternion[0]
    r1 = element.Quaternion[1]
    r2 = element.Quaternion[2]
    r3 = element.Quaternion[3]

    T_tmp = np.array([
            [r0 ** 2 + r1 ** 2 - r2 ** 2 - r3 ** 2, 2 * (r1 * r2 - r0 * r3), 2 * (r1 * r3 + r0 * r2)],
            [2 * (r1 * r2 + r0 * r3), r0 ** 2 - r1 ** 2 + r2 ** 2 - r3 ** 2, 2 * (r2 * r3 - r0 * r1)],
            [2 * (r3 * r1 - r0 * r2), 2 * (r3 * r2 + r0 * r1), r0 ** 2 - r1 ** 2 - r2 ** 2 + r3 ** 2],

    ])

    nx = T_tmp[:, 0]
    ny = T_tmp[:, 1]
    nz = T_tmp[:, 2]

    delta_x = new_coords[3:6] - new_coords[0:3]
    delta_x /= np.linalg.norm(delta_x)
    n = nx + delta_x
    n /= np.linalg.norm(n)
    n_xyz = np.array([-nx, ny, nz]).T
    tmp = np.outer(n, np.transpose(n))
    tmp = (np.identity(3) - 2 * tmp)
    T_sol = np.dot(tmp, n_xyz)
    T = element.LocalRotationMatrix

    try:
        assert (T - T_sol < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in local transformation matrix calculation\n"
        msg += "Quaternion: " + str(element.Quaternion)
        msg += "Reference Rotation Matrix:\n" + str(element.LocalReferenceRotationMatrix)
        msg += "T is suppose to be:\n" + str(T_sol)
        msg += "\nIt is however:\n" + str(T)
        print(msg)

    dv = element.v
    # Eq.(4.84) Klaus, with relate to the local axis
    dv_sol = np.dot(S.T, dp)

    try:
        assert (dv - dv_sol < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in deformation mode calculation\n"
        msg += "T is suppose to be:\n" + str(dv_sol)
        msg += "\nIt is however:\n" + str(dv)
        print(msg)

    ke_mat_2 = element._get_element_stiffness_matrix_material()

    K = element.get_element_stiffness_matrix()
    M = element.get_element_mass_matrix()

    try:
        assert (ke_mat_1 - ke_mat_2 < TOL).all()
    except AssertionError:
        print("Ke_const wrong")

    f_test = np.dot(K, dp)
    err = f_test - element.nodal_force_global
    np.set_printoptions(precision=1)
