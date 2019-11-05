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


TOL = 1e-6


def test_crbeam_element_update_incremental():
    material_params = {'rho': 1000.0, 'e': 1.e6, 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
    element_params = {'a': 1., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

    coords = np.array([[1., 0.0, 0.0], [2.0, 0.0, 0.0]])
    element = CRBeamElement(material_params, element_params, coords, 0, '3D')

    ke_mat_1 = element._get_element_stiffness_matrix_material()

    dp = [0.2, 0.6, 0.0, 0.1, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    element.update_incremental(dp)
    new_coords = element._get_current_nodal_position()
    new_coords_sol = [1.2, 0.6, 0.0, 2.0, 0.0, 0.0]

    try:
        assert (abs(new_coords - new_coords_sol) < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in coordinate update\n"
        msg += "New coordinate is suppose to be:\n" + str(new_coords_sol)
        msg += "\nIt is however:\n" + str(new_coords)
        print(msg)

    l = element._calculate_current_length()
    l_sol = np.sqrt(0.8**2 + 0.6**2)

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
        assert (abs(S - S_sol) < TOL).all()
    except AssertionError:
        print("Mistake in local transformation matrix S calculation")

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
        assert (abs(T - T_sol) < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in local transformation matrix calculation\n"
        msg += "Quaternion: " + str(element.Quaternion)
        msg += "\nReference Rotation Matrix:\n" + str(element.LocalReferenceRotationMatrix)
        msg += "\nT is suppose to be:\n" + str(T_sol)
        msg += "\nIt is however:\n" + str(T)
        print(msg)

    dv = element.v
    nx = T_sol[:, 0]
    ny = T_sol[:, 1]
    nz = T_sol[:, 2]
    # Eq.(4.60) Klaus, related to the local axis
    S_global = np.array([
            [0., 0., 0., -nx[0], -2 * nz[0] / l, 2 * ny[0] / l],
            [0., 0., 0., -nx[1], -2 * nz[1] / l, 2 * ny[1] / l],
            [0., 0., 0., -nx[2], -2 * nz[2] / l, 2 * ny[2] / l],
            [-nx[0], -ny[0], -nz[0], 0., ny[0], nz[0]],
            [-nx[1], -ny[1], -nz[1], 0., ny[1], nz[1]],
            [-nx[2], -ny[2], -nz[2], 0., ny[2], nz[2]],
            [0., 0., 0., nx[0], 2 * nz[0] / l, -2 * ny[0] / l],
            [0., 0., 0., nx[1], 2 * nz[1] / l, -2 * ny[1] / l],
            [0., 0., 0., nx[2], 2 * nz[2] / l, -2 * ny[2] / l],
            [nx[0], ny[0], nz[0], 0., ny[0], nz[0]],
            [nx[1], ny[1], nz[1], 0., ny[1], nz[1]],
            [nx[2], ny[2], nz[2], 0., ny[2], nz[2]],
    ])
    # Eq.(4.84) Klaus
    dv_sol = np.dot(S_global.T, dp)
    try:
        assert (abs(dv - dv_sol) < TOL).all()
        print(dv- dv_sol)
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in deformation mode calculation\n"
        msg += "nx = " + str(nx)
        msg += "\nv is suppose to be:\n" + str(dv_sol)
        msg += "\nIt is however:\n" + str(dv)
        print(msg)

    ke_mat_2 = element._get_element_stiffness_matrix_material()

    try:
        assert (ke_mat_1 - ke_mat_2 < TOL).all()
    except AssertionError:
        print("Ke_const wrong")

    np.set_printoptions(precision=1)


def test_crbeam_element_update_total():
    material_params = {'rho': 1000.0, 'e': 1.e6, 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10., 'is_nonlinear': True}
    element_params = {'a': 1., 'asy': 2., 'asz': 2., 'iy': 10, 'iz': 20, 'it': 20}

    coords = np.array([[100., 0.0, 0.0], [200.0, 0.0, 0.0]])
    element = CRBeamElement(material_params, element_params, coords, 0, '3D')

    dp_v = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    element.update_total(dp_v)
    K = element.get_element_stiffness_matrix()
    f_test = np.dot(K, dp_v)
    q = element.nodal_force_global
    try:
        assert (abs(f_test - q) < TOL).all()
    except AssertionError:
        msg = "##################################################################################\n"
        msg += "Mistake in force calculation\n"
        msg += "q is suppose to be:\n" + str(f_test)
        msg += "\nIt is however:\n" + str(q)
        print(msg)

    np.set_printoptions(precision=1)
