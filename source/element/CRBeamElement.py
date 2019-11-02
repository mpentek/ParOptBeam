import numpy as np
import sys

from source.element.Element import Element

EPSILON = sys.float_info.epsilon


def rotate_vector(quaternion, vector):
    """
    Rotates a vector using this quaternion.
    Note: this is faster than constructing the rotation matrix and perform the matrix
    multiplication for a single vector.
    :param quaternion:
    :param vector: the input source vector - rotated on exit
    :return: vector
    """
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    # b = 2.0 (quaternion x vector)
    b0 = 2.0 * (y * vector[2] - z * vector[1])
    b1 = 2.0 * (z * vector[0] - x * vector[2])
    b2 = 2.0 * (x * vector[1] - y * vector[0])

    # c = 2.0 (quaternion x b)
    c0 = y * b2 - z * b1
    c1 = z * b0 - x * b2
    c2 = x * b1 - y * b0

    vector[0] += b0 * w + c0
    vector[1] += b1 * w + c1
    vector[2] += b2 * w + c2

    return vector


def apply_transformation(TransformationMatrix, M):
    # transformation M = T * M * trans(T)
    aux_matrix = np.matmul(TransformationMatrix, M)
    M_transformed = np.matmul(aux_matrix, np.transpose(TransformationMatrix))
    return M_transformed


class CRBeamElement(Element):
    def __init__(self, material_params, element_params, nodal_coords, index, domain_size):
        if domain_size == '2D':
            err_msg = "2D CR-Beam element not implemented"
            err_msg += "Please use only 3D CR-Beam element"
            raise Exception(err_msg)

        super().__init__(material_params, element_params, nodal_coords, index, domain_size)

        self.evaluate_relative_importance_of_shear()

        self.Iteration = 0

        # transformation matrix T = [nx0, ny0, nz0]
        # transformation matrix T = [nx, ny, nz]
        self.LocalRotationMatrix = np.zeros([self.Dimension, self.Dimension])
        # transformation matrix
        self.TransformationMatrix = np.zeros([self.ElementSize, self.ElementSize])

        # for calculating deformation
        self._QuaternionVEC_A = np.zeros(self.Dimension)
        self._QuaternionVEC_B = np.zeros(self.Dimension)
        self._QuaternionSCA_A = 1.0
        self._QuaternionSCA_B = 1.0

        # initializing transformation matrix for iteration = 0
        self._update_rotation_matrix_local()
        self.TransformationMatrix = self._calculate_initial_local_cs()

        # initializing bisector and vector_difference for calculating phi_a and phi_s
        self.Bisectrix = np.zeros(self.Dimension)
        self.VectorDifference = np.zeros(self.Dimension)

        self._print_element_information()

    def _print_element_information(self):
        if self.isNonlinear:
            msg = "Nonlinear "
        else:
            msg = "Linear "
        msg += str(self.domain_size) + " Co-Rotational Beam Element " + str(self.index) + "\n"
        msg += "Initial coordinates: \n"
        msg += str(self.ReferenceCoords[:3]) + "\n"
        msg += str(self.ReferenceCoords[3:]) + "\n"
        msg += "L: " + str(self.L) + "\n"
        msg += "A: " + str(self.A) + "\n"
        msg += "Asy: " + str(self.Asy) + "\n"
        msg += "Asz: " + str(self.Asz) + "\n"
        msg += "Iy: " + str(self.Iy) + "\n"
        msg += "Iz: " + str(self.Iz) + "\n"
        print(msg)

    def update_internal_force(self):
        self.Iteration += 1
        self._calculate_transformation_matrix()
        # update local nodal force
        self._calculate_local_nodal_forces()
        self.nodal_force_global = np.dot(self.TransformationMatrix, self.nodal_force_local)

    def get_element_mass_matrix(self):
        MassMatrix = self._get_consistent_mass_matrix()
        TransformedMassMatrix = apply_transformation(self.TransformationMatrix, MassMatrix)
        return TransformedMassMatrix

    def _get_consistent_mass_matrix(self):
        """
            there are two options for mass matrix calculation, either the consistent mass matrix or the lumped mass matrix
            here the consistent mass matrix is calculated because the lumped mass matrix is singular for beam elements
        """
        MassMatrix = np.zeros([self.ElementSize, self.ElementSize])
        L2 = self.L * self.L
        Phiz = (12.0 * self.E * self.Iz) / (L2 * self.G * self.Asy)
        Phiy = (12.0 * self.E * self.Iy) / (L2 * self.G * self.Asz)

        # rotational inertia
        IRy = self.Iy
        IRz = self.Iz

        CTy = (self.rho * self.A * self.L) / ((1 + Phiy) * (1 + Phiy))
        CTz = (self.rho * self.A * self.L) / ((1 + Phiz) * (1 + Phiz))
        CRy = (self.rho * IRy) / ((1 + Phiy) * (1 + Phiy) * self.L)
        CRz = (self.rho * IRz) / ((1 + Phiz) * (1 + Phiz) * self.L)

        # longitudinal forces + torsional moment
        M00 = (1.0 / 3.0) * self.A * self.rho * self.L
        M06 = M00 / 2.0
        M33 = (self.It * self.L * self.rho) / 3.0
        M39 = M33 / 2.0

        MassMatrix[0, 0] = M00
        MassMatrix[0, 6] = M06
        MassMatrix[6, 6] = M00
        MassMatrix[3, 3] = M33
        MassMatrix[3, 9] = M39
        MassMatrix[9, 9] = M33

        temp_bending_mass_matrix = self.build_single_mass_matrix(Phiz, CTz, CRz, self.L, +1)

        MassMatrix[1, 1] = temp_bending_mass_matrix[0, 0]
        MassMatrix[1, 5] = temp_bending_mass_matrix[0, 1]
        MassMatrix[1, 7] = temp_bending_mass_matrix[0, 2]
        MassMatrix[1, 11] = temp_bending_mass_matrix[0, 3]
        MassMatrix[5, 5] = temp_bending_mass_matrix[1, 1]
        MassMatrix[5, 7] = temp_bending_mass_matrix[1, 2]
        MassMatrix[5, 11] = temp_bending_mass_matrix[1, 3]
        MassMatrix[7, 7] = temp_bending_mass_matrix[2, 2]
        MassMatrix[7, 11] = temp_bending_mass_matrix[2, 3]
        MassMatrix[11, 11] = temp_bending_mass_matrix[3, 3]

        temp_bending_mass_matrix = self.build_single_mass_matrix(Phiy, CTy, CRy, self.L, -1)

        MassMatrix[2, 2] = temp_bending_mass_matrix[0, 0]
        MassMatrix[2, 4] = temp_bending_mass_matrix[0, 1]
        MassMatrix[2, 8] = temp_bending_mass_matrix[0, 2]
        MassMatrix[2, 10] = temp_bending_mass_matrix[0, 3]
        MassMatrix[4, 4] = temp_bending_mass_matrix[1, 1]
        MassMatrix[4, 8] = temp_bending_mass_matrix[1, 2]
        MassMatrix[4, 10] = temp_bending_mass_matrix[1, 3]
        MassMatrix[8, 8] = temp_bending_mass_matrix[2, 2]
        MassMatrix[8, 10] = temp_bending_mass_matrix[2, 3]
        MassMatrix[10, 10] = temp_bending_mass_matrix[3, 3]

        for i in range(self.ElementSize):
            for j in range(i):
                MassMatrix[j, i] = MassMatrix[i, j]

        return MassMatrix

    def build_single_mass_matrix(self, Phi, CT, CR, L, dir):
        MatSize = self.NumberOfNodes * 2
        mass_matrix = np.zeros([MatSize, MatSize])
        temp_mass_matrix = np.zeros([MatSize, MatSize])

        Phi2 = Phi * Phi
        L2 = L * L

        temp_mass_matrix[0, 0] = (13.0 / 35.0) + (7.0 / 10.0) * Phi + (1.0 / 3.0) * Phi2
        temp_mass_matrix[0, 1] = dir * ((11.0 / 210.0) + (11.0 / 210.0) * Phi + (1.0 / 24.0) * Phi2) * L
        temp_mass_matrix[0, 2] = (9.0 / 70.0) + (3.0 / 10.0) * Phi + (1.0 / 6.0) * Phi2
        temp_mass_matrix[0, 3] = -((13.0 / 420.0) + (3.0 / 40.0) * Phi + (1.0 / 24.0) * Phi2) * L * dir

        temp_mass_matrix[1, 0] = temp_mass_matrix[0, 1]
        temp_mass_matrix[1, 1] = ((1.0 / 105.0) + (1.0 / 60.0) * Phi + (1.0 / 120.0) * Phi2) * L2

        temp_mass_matrix[1, 2] = dir * ((13.0 / 420.0) + (3.0 / 40.0) * Phi + (1.0 / 24.0) * Phi2) * L
        temp_mass_matrix[1, 3] = -((1.0 / 140.0) + (1.0 / 60.0) * Phi + (1.0 / 120.0) * Phi2) * L2
        temp_mass_matrix[2, 0] = temp_mass_matrix[0, 2]
        temp_mass_matrix[2, 1] = temp_mass_matrix[1, 2]
        temp_mass_matrix[2, 2] = (13.0 / 35.0) + (7.0 / 10.0) * Phi + (1.0 / 3.0) * Phi2
        temp_mass_matrix[2, 3] = -((11.0 / 210.0) + (11.0 / 210.0) * Phi + (1.0 / 24.0) * Phi2) * L * dir
        temp_mass_matrix[3, 0] = temp_mass_matrix[0, 3]
        temp_mass_matrix[3, 1] = temp_mass_matrix[1, 3]
        temp_mass_matrix[3, 2] = temp_mass_matrix[2, 3]
        temp_mass_matrix[3, 3] = ((1.0 / 105.0) + (1.0 / 60.0) * Phi + (1.0 / 120.0) * Phi2) * L2

        temp_mass_matrix *= CT
        mass_matrix += temp_mass_matrix
        temp_mass_matrix = np.zeros([MatSize, MatSize])

        temp_mass_matrix[0, 0] = 6.0 / 5.0
        temp_mass_matrix[0, 1] = dir * ((1.0 / 10.0) - (1.0 / 2.0) * Phi) * L
        temp_mass_matrix[0, 2] = -6.0 / 5.0
        temp_mass_matrix[0, 3] = dir * ((1.0 / 10.0) - (1.0 / 2.0) * Phi) * L
        temp_mass_matrix[1, 0] = temp_mass_matrix[0, 1]
        temp_mass_matrix[1, 1] = ((2.0 / 15.0) + (1.0 / 6.0) * Phi + (1.0 / 3.0) * Phi2) * L2
        temp_mass_matrix[1, 2] = dir * ((-1.0 / 10.0) + (1.0 / 2.0) * Phi) * L
        temp_mass_matrix[1, 3] = -((1.0 / 30.0) + (1.0 / 6.0) * Phi - (1.0 / 6.0) * Phi2) * L2
        temp_mass_matrix[2, 0] = temp_mass_matrix[0, 2]
        temp_mass_matrix[2, 1] = temp_mass_matrix[1, 2]
        temp_mass_matrix[2, 2] = 6.0 / 5.0
        temp_mass_matrix[2, 3] = dir * ((-1.0 / 10.0) + (1.0 / 2.0) * Phi) * L
        temp_mass_matrix[3, 0] = temp_mass_matrix[0, 3]
        temp_mass_matrix[3, 1] = temp_mass_matrix[1, 3]
        temp_mass_matrix[3, 2] = temp_mass_matrix[2, 3]
        temp_mass_matrix[3, 3] = ((2.0 / 15.0) + (1.0 / 6.0) * Phi + (1.0 / 3.0) * Phi2) * L2

        temp_mass_matrix *= CR
        mass_matrix += temp_mass_matrix

        return mass_matrix

    def get_element_stiffness_matrix(self):
        # calculate local nodal forces
        self._calculate_local_nodal_forces()

        # resizing the matrices + create memory for LHS
        Ke = np.zeros([self.ElementSize, self.ElementSize])
        # creating LHS
        Ke += self._get_element_stiffness_matrix_material()
        Ke += self._get_element_stiffness_matrix_geometry()

        TransformedStiffnessMatrix = apply_transformation(self.TransformationMatrix, Ke)
        return TransformedStiffnessMatrix

    def _get_element_stiffness_matrix_material(self):
        """
            elastic part of the total stiffness matrix
        """
        L = self.L

        # shear coefficients
        Psi_y = self._calculate_psi(self.Iy, self.Asz)
        Psi_z = self._calculate_psi(self.Iz, self.Asy)

        ke_const = np.zeros([self.ElementSize, self.ElementSize])

        self.L3 = L * L * L
        self.L2 = L * L

        ke_const[0, 0] = self.E * self.A / L
        ke_const[6, 0] = -1.0 * ke_const[0, 0]
        ke_const[0, 6] = ke_const[6, 0]
        ke_const[6, 6] = ke_const[0, 0]

        ke_const[1, 1] = 12.0 * self.E * self.Iz * Psi_z / self.L3
        ke_const[1, 7] = -1.0 * ke_const[1, 1]
        ke_const[1, 5] = 6.0 * self.E * self.Iz * Psi_z / self.L2
        ke_const[1, 11] = ke_const[1, 5]

        ke_const[2, 2] = 12.0 * self.E * self.Iy * Psi_y / self.L3
        ke_const[2, 8] = -1.0 * ke_const[2, 2]
        ke_const[2, 4] = -6.0 * self.E * self.Iy * Psi_y / self.L2
        ke_const[2, 10] = ke_const[2, 4]

        ke_const[4, 2] = ke_const[2, 4]
        ke_const[5, 1] = ke_const[1, 5]
        ke_const[3, 3] = self.G * self.It / L
        ke_const[4, 4] = self.E * self.Iy * (3.0 * Psi_y + 1.0) / L
        ke_const[5, 5] = self.E * self.Iz * (3.0 * Psi_z + 1.0) / L
        ke_const[4, 8] = -1.0 * ke_const[4, 2]
        ke_const[5, 7] = -1.0 * ke_const[5, 1]
        ke_const[3, 9] = -1.0 * ke_const[3, 3]
        ke_const[4, 10] = self.E * self.Iy * (3.0 * Psi_y - 1.0) / L
        ke_const[5, 11] = self.E * self.Iz * (3.0 * Psi_z - 1.0) / L

        ke_const[7, 1] = ke_const[1, 7]
        ke_const[7, 5] = ke_const[5, 7]
        ke_const[7, 7] = ke_const[1, 1]
        ke_const[7, 11] = ke_const[7, 5]

        ke_const[8, 2] = ke_const[2, 8]
        ke_const[8, 4] = ke_const[4, 8]
        ke_const[8, 8] = ke_const[2, 2]
        ke_const[8, 10] = ke_const[8, 4]

        ke_const[9, 3] = ke_const[3, 9]
        ke_const[9, 9] = ke_const[3, 3]

        ke_const[10, 2] = ke_const[2, 10]
        ke_const[10, 4] = ke_const[4, 10]
        ke_const[10, 8] = ke_const[8, 10]
        ke_const[10, 10] = ke_const[4, 4]

        ke_const[11, 1] = ke_const[1, 11]
        ke_const[11, 5] = ke_const[5, 11]
        ke_const[11, 7] = ke_const[7, 11]
        ke_const[11, 11] = ke_const[5, 5]

        return ke_const

    def _get_element_stiffness_matrix_geometry(self):
        """
            geometric part of the total stiffness matrix
        """

        N = self.nodal_force_local[6]
        Mt = self.nodal_force_local[9]
        my_A = self.nodal_force_local[4]
        mz_A = self.nodal_force_local[5]
        my_B = self.nodal_force_local[10]
        mz_B = self.nodal_force_local[11]

        L = self._calculate_current_length()
        Qy = -1.0 * (mz_A + mz_B) / L
        Qz = (my_A + my_B) / L

        ke_geo = np.zeros([self.ElementSize, self.ElementSize])

        ke_geo[0, 1] = -Qy / L
        ke_geo[0, 2] = -Qz / L
        ke_geo[0, 7] = -1.0 * ke_geo[0, 1]
        ke_geo[0, 8] = -1.0 * ke_geo[0, 2]

        ke_geo[1, 0] = ke_geo[0, 1]

        ke_geo[1, 1] = 1.2 * N / L

        ke_geo[1, 3] = my_A / L
        ke_geo[1, 4] = Mt / L

        ke_geo[1, 5] = N / 10.0

        ke_geo[1, 6] = ke_geo[0, 7]
        ke_geo[1, 7] = -1.0 * ke_geo[1, 1]
        ke_geo[1, 9] = my_B / L
        ke_geo[1, 10] = -1.0 * ke_geo[1, 4]
        ke_geo[1, 11] = ke_geo[1, 5]

        ke_geo[2, 0] = ke_geo[0, 2]
        ke_geo[2, 2] = ke_geo[1, 1]
        ke_geo[2, 3] = mz_A / L
        ke_geo[2, 4] = -1.0 * ke_geo[1, 5]
        ke_geo[2, 5] = ke_geo[1, 4]
        ke_geo[2, 6] = ke_geo[0, 8]
        ke_geo[2, 8] = ke_geo[1, 7]
        ke_geo[2, 9] = mz_B / L
        ke_geo[2, 10] = ke_geo[2, 4]
        ke_geo[2, 11] = ke_geo[1, 10]

        for i in range(3):
            ke_geo[3, i] = ke_geo[i, 3]

        ke_geo[3, 4] = (-mz_A / 3.0) + (mz_B / 6.0)
        ke_geo[3, 5] = (my_A / 3.0) - (my_B / 6.0)
        ke_geo[3, 7] = -my_A / L
        ke_geo[3, 8] = -mz_A / L
        ke_geo[3, 10] = L * Qy / 6.0
        ke_geo[3, 11] = L * Qz / 6.0

        for i in range(4):
            ke_geo[4, i] = ke_geo[i, 4]

        ke_geo[4, 4] = 2.0 * L * N / 15.0
        ke_geo[4, 7] = -Mt / L
        ke_geo[4, 8] = N / 10.0
        ke_geo[4, 9] = ke_geo[3, 10]
        ke_geo[4, 10] = -L * N / 30.0
        ke_geo[4, 11] = Mt / 2.0

        for i in range(5):
            ke_geo[5, i] = ke_geo[i, 5]

        ke_geo[5, 5] = ke_geo[4, 4]
        ke_geo[5, 7] = -N / 10.0
        ke_geo[5, 8] = -Mt / L
        ke_geo[5, 9] = ke_geo[3, 11]
        ke_geo[5, 10] = -1.0 * ke_geo[4, 11]
        ke_geo[5, 11] = ke_geo[4, 10]

        for i in range(6):
            ke_geo[6, i] = ke_geo[i, 6]

        ke_geo[6, 7] = ke_geo[0, 1]
        ke_geo[6, 8] = ke_geo[0, 2]

        for i in range(7):
            ke_geo[7, i] = ke_geo[i, 7]

        ke_geo[7, 7] = ke_geo[1, 1]
        ke_geo[7, 9] = -1.0 * ke_geo[1, 9]
        ke_geo[7, 10] = ke_geo[4, 1]
        ke_geo[7, 11] = ke_geo[2, 4]

        for i in range(8):
            ke_geo[8, i] = ke_geo[i, 8]

        ke_geo[8, 8] = ke_geo[1, 1]
        ke_geo[8, 9] = -1.0 * ke_geo[2, 9]
        ke_geo[8, 10] = ke_geo[1, 5]
        ke_geo[8, 11] = ke_geo[1, 4]

        for i in range(9):
            ke_geo[9, i] = ke_geo[i, 9]

        ke_geo[9, 10] = (mz_A / 6.0) - (mz_B / 3.0)
        ke_geo[9, 11] = (-my_A / 6.0) + (my_B / 3.0)

        for i in range(10):
            ke_geo[10, i] = ke_geo[i, 10]

        ke_geo[10, 10] = ke_geo[4, 4]

        for i in range(11):
            ke_geo[11, i] = ke_geo[i, 11]

        ke_geo[11, 11] = ke_geo[4, 4]

        return ke_geo

    def _calculate_deformation_stiffness(self):
        L = self.L
        Psi_y = self._calculate_psi(self.Iy, self.Asz)
        Psi_z = self._calculate_psi(self.Iz, self.Asy)

        Kd = np.zeros([self.LocalSize, self.LocalSize])

        Kd[0, 0] = self.G * self.It / L
        Kd[1, 1] = self.E * self.Iy / L
        Kd[2, 2] = self.E * self.Iz / L
        Kd[3, 3] = self.E * self.A / L
        Kd[4, 4] = 3.0 * self.E * self.Iy * Psi_y / L
        Kd[5, 5] = 3.0 * self.E * self.Iz * Psi_z / L

        l = self._calculate_current_length()
        N = self.nodal_force_local[6]
        Qy = -1.0 * (self.nodal_force_local[5] + self.nodal_force_local[11]) / l
        Qz = 1.0 * (self.nodal_force_local[4] + self.nodal_force_local[10]) / l

        N1 = l * N / 12.0
        N2 = l * N / 20.0
        Qy1 = -l * Qy / 6.0
        Qz1 = -l * Qz / 6.0

        Kd[1, 1] += N1
        Kd[2, 2] += N1
        Kd[4, 4] += N2
        Kd[5, 5] += N2

        Kd[0, 1] += Qy1
        Kd[0, 2] += Qz1
        Kd[1, 0] += Qy1
        Kd[2, 0] += Qz1

        return Kd

    def _calculate_psi(self, I, A_eff):
        # TODO: check which length to use, paper and implementation different
        phi = (12.0 * self.E * I) / (L * L * self.G * A_eff)

        # interpret input A_eff == 0 as shear stiff -> psi = 1.0
        if A_eff == 0.0:
            psi = 1.0
        else:
            psi = 1.0 / (1.0 + phi)
        return psi

    def _calculate_transformation_s(self):
        S = np.zeros([self.ElementSize, self.LocalSize])
        L = self._calculate_current_length()

        S[0, 3] = -1.0
        S[1, 5] = 2.0 / L
        S[2, 4] = -2.0 / L
        S[3, 0] = -1.0
        S[4, 1] = -1.0
        S[4, 4] = 1.0
        S[5, 2] = -1.0
        S[5, 5] = 1.0
        S[6, 3] = 1.0
        S[7, 5] = -2.0 / L
        S[8, 4] = 2.0 / L
        S[9, 0] = 1.0
        S[10, 1] = 1.0
        S[10, 4] = 1.0
        S[11, 2] = 1.0
        S[11, 5] = 1.0

        return S

    def _calculate_local_nodal_forces(self):

        # element force t
        element_forces_t = self._calculate_element_forces()

        # updating transformation matrix S
        S = self._calculate_transformation_s()
        self.nodal_force_local = np.dot(S, element_forces_t)

    def _calculate_element_forces(self):
        # reference length
        L = self.L
        # current length
        l = self._calculate_current_length()
        # symmetric deformation mode
        phi_s = self._calculate_symmetric_deformation_mode()
        # asymmetric deformation mode
        phi_a = self._calculate_antisymmetric_deformation_mode()

        deformation_modes_total_v = np.zeros(self.LocalSize)
        deformation_modes_total_v[3] = l - L

        for i in range(3):
            deformation_modes_total_v[i] = phi_s[i]
        for i in range(2):
            deformation_modes_total_v[i + 4] = phi_a[i + 1]

        Kd = self._calculate_deformation_stiffness()
        element_forces_t = np.dot(Kd, deformation_modes_total_v)
        return element_forces_t

    def _calculate_symmetric_deformation_mode(self):
        """
            this function calculates symmetric part of vector v
            reference: Eq. (4.53) Klaus
        :return: phi_s
        """
        return phi_s

    def _calculate_antisymmetric_deformation_mode(self):
        """
            this function calculates anti-symmetric part of v
            reference: Eq. (4.54) Klaus
        :return: phi_a
        """
            phi_a = np.dot((np.transpose(self.LocalRotationMatrix)), temp_vector)
            phi_a *= 4.0
        return phi_a

    def _update_rotation_matrix_local(self):
        """
        this function updates the local transformation matrix T = [nx, ny, nz]
        Check reference in [p.134]:
            Steen Krenk. Non-linear modeling and analysis of solids and structures.
            Cambridge Univ. Press, 2009.
        """
        increment_deformation = self._update_increment_deformation()
        for i in range(0, self.Dimension):
            d_phi_a[i] = increment_deformation[i + 3]
            d_phi_b[i] = increment_deformation[i + 9]

        # calculating quaternions
        drA_vec = 0.50 * d_phi_a
        drB_vec = 0.50 * d_phi_b

        drA_sca = 0.0
        drB_sca = 0.0

        for i in range(0, self.Dimension):
            drA_sca += drA_vec[i] * drA_vec[i]
            drB_sca += drB_vec[i] * drB_vec[i]

        drA_sca = np.sqrt(1.0 - drA_sca)
        drB_sca = np.sqrt(1.0 - drB_sca)

        # Node A
        temp_vec = self._QuaternionVEC_A
        temp_scalar = self._QuaternionSCA_A
        self._QuaternionVEC_A = drA_sca * temp_scalar
        for i in range(self.Dimension):
            self._QuaternionSCA_A -= drA_vec[i] * temp_vec[i]

        self._QuaternionVEC_A = drA_sca * temp_vec
        self._QuaternionVEC_A += temp_scalar * drA_vec
        self._QuaternionVEC_A += np.cross(drA_vec, temp_vec)

        # Node B
        temp_vec = self._QuaternionVEC_B
        temp_scalar = self._QuaternionSCA_B
        self._QuaternionVEC_B = drB_sca * temp_scalar
        for i in range(self.Dimension):
            self._QuaternionSCA_B -= drB_vec[i] * temp_vec[i]

        self._QuaternionVEC_B = drB_sca * temp_vec
        self._QuaternionVEC_B += temp_scalar * drB_vec
        self._QuaternionVEC_B += np.cross(drB_vec, temp_vec)

        # scalar part of difference quaternion
        scalar_diff = (self._QuaternionSCA_A + self._QuaternionSCA_B) * (self._QuaternionSCA_A + self._QuaternionSCA_B)
        temp_vec = self._QuaternionVEC_A + self._QuaternionVEC_B
        scalar_diff += np.linalg.norm(temp_vec) * np.linalg.norm(temp_vec)
        scalar_diff = 0.5 * np.sqrt(scalar_diff)

        # mean rotation quaternion
        mean_rotation_scalar = (self._QuaternionSCA_A + self._QuaternionSCA_B) * 0.50 / scalar_diff
        mean_rotation_vector = (self._QuaternionVEC_A + self._QuaternionVEC_B) * 0.50 / scalar_diff

        # vector part of difference quaternion
        VectorDifferences = self._QuaternionSCA_A * self._QuaternionVEC_B
        VectorDifferences -= self._QuaternionSCA_A * self._QuaternionVEC_A
        VectorDifferences += np.cross(self._QuaternionVEC_A, self._QuaternionVEC_B)

        VectorDifferences = 0.5 * VectorDifferences / scalar_diff
        # rotate initial element basis
        r0 = mean_rotation_scalar
        r1 = mean_rotation_vector[0]
        r2 = mean_rotation_vector[1]
        r3 = mean_rotation_vector[2]
        reference_transformation = self._calculate_initial_local_cs()
        rotated_nx0 = np.zeros(self.Dimension)
        rotated_ny0 = np.zeros(self.Dimension)
        rotated_nz0 = np.zeros(self.Dimension)

        for i in range(self.Dimension):
            rotated_nx0[i] = reference_transformation[i, 0]
            rotated_ny0[i] = reference_transformation[i, 1]
            rotated_nz0[i] = reference_transformation[i, 2]

        quaternion = [r0, r1, r2, r3]
        rotated_nx0 = rotate_vector(quaternion, rotated_nx0)
        rotated_ny0 = rotate_vector(quaternion, rotated_ny0)
        rotated_nz0 = rotate_vector(quaternion, rotated_nz0)

        rotated_coordinate_system = np.zeros([self.Dimension, self.Dimension])
        rotated_coordinate_system[:, 0] = rotated_nx0
        rotated_coordinate_system[:, 1] = rotated_ny0
        rotated_coordinate_system[:, 2] = rotated_nz0

        CurrentCoords = self._get_current_nodal_position()

        # rotate basis to element axis + redefine R
        delta_x = np.zeros(self.Dimension)
        for i in range(self.Dimension):
            delta_x[i] = CurrentCoords[self.Dimension + i] - CurrentCoords[i]
        vector_norm = np.linalg.norm(delta_x)

        if vector_norm > EPSILON:
            delta_x /= vector_norm

        Bisectrix = rotated_nx0 + delta_x
        vector_norm = np.linalg.norm(Bisectrix)

        if vector_norm > EPSILON:
            Bisectrix /= vector_norm

        n_xyz = np.zeros([self.Dimension, self.Dimension])

        n_xyz[:, 0] = -rotated_coordinate_system[:, 0]
        n_xyz[:, 1] = rotated_coordinate_system[:, 1]
        n_xyz[:, 2] = rotated_coordinate_system[:, 2]

        Identity = np.identity(self.Dimension)
        Identity -= 2.0 * np.outer(Bisectrix, Bisectrix)
        n_xyz = np.matmul(Identity, n_xyz)
        self.LocalRotationMatrix = n_xyz
        self.Bisectrix = Bisectrix
        self.VectorDifference = VectorDifferences

    def _calculate_transformation_matrix(self):
        """
        This function calculates the transformation matrix to globalize/localize vectors and/or matrices
        """
        # update local CS
        self._update_rotation_matrix_local()
        # Building the rotation matrix for the local element matrix
        self.TransformationMatrix = self._assemble_small_in_big_matrix(self.LocalRotationMatrix)

    def _assemble_small_in_big_matrix(self, small_matrix):
        numerical_limit = EPSILON
        big_matrix = np.zeros([self.ElementSize, self.ElementSize])

        for k in range(0, self.ElementSize, self.Dimension):
            for i in range(self.Dimension):
                for j in range(self.Dimension):
                    if abs(small_matrix[i, j]) <= numerical_limit:
                        big_matrix[i + k, j + k] = 0.0
                    else:
                        big_matrix[i + k, j + k] = small_matrix[i, j]
        return big_matrix

    def _calculate_initial_local_cs(self):
        direction_vector_x = np.zeros(self.Dimension)

        for i in range(self.Dimension):
            direction_vector_x[i] = (self.ReferenceCoords[i + self.Dimension] - self.ReferenceCoords[i])

        temp_matrix = np.zeros([self.Dimension, self.Dimension])
        # no user defined local axis 2 input available
        theta_custom = 0.0
        global_z = np.zeros(self.Dimension)
        global_z[2] = 1.0

        v2 = np.zeros(self.Dimension)
        v3 = np.zeros(self.Dimension)

        vector_norm = np.linalg.norm(direction_vector_x)
        if vector_norm > EPSILON:
            direction_vector_x /= vector_norm

        if np.linalg.norm(direction_vector_x[2] - 1.0) < EPSILON:
            v2[1] = 1.0
            v3[0] = -1.0
        elif np.linalg.norm(direction_vector_x[2] + 1.0) < EPSILON:
            v2[1] = 1.0
            v3[0] = 1.0
        else:
            v2 = np.cross(global_z, direction_vector_x)
            v3 = np.cross(direction_vector_x, v2)

        # manual rotation around the beam axis
        if np.linalg.norm(theta_custom) > EPSILON:
            nz_temp = v3
            ny_temp = v2
            cos_theta = np.cos(theta_custom)
            sin_theta = np.sin(theta_custom)

            v2 = ny_temp * cos_theta + nz_temp * sin_theta
            vector_norm = np.linalg.norm(v2)

            if vector_norm > EPSILON:
                v2 /= vector_norm

            v3 = nz_temp * cos_theta - ny_temp * sin_theta
            vector_norm = np.linalg.norm(v3)

            if vector_norm > EPSILON:
                v3 /= vector_norm

        for i in range(self.Dimension):
            temp_matrix[i, 0] = direction_vector_x[i]
            temp_matrix[i, 1] = v2[i]
            temp_matrix[i, 2] = v3[i]

        reference_transformation = self._assemble_small_in_big_matrix(temp_matrix)

        return reference_transformation
