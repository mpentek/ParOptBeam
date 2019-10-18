import numpy as np

from source.element.Element import Element


class CRBeamElement(Element):
    def __init__(self, parameters, domain_size):
        super().__init__(parameters, domain_size)

        # material properties
        self.E = self.parameters['e']
        self.rho = self.parameters['rho']
        self.nu = self.parameters['nu']
        self.G = self.parameters['g'] = self.E / 2 / (1 + self.nu)

        # area
        self.A = None
        # effective area of shear
        self.Asy = None
        self.Asz = None

        # length of one element - assuming an equidistant grid
        self.Li = self.parameters['lx_i']

        # second moment of inertia
        self.Iy = None
        self.Iz = None
        # torsion constant J
        self.It = None
        # evaluating torsional inertia
        self.Ip = None

        self.Py = None
        self.Pz = None

        # element properties
        NumberOfNodes = 2
        Dimension = 3
        self.LocalSize = NumberOfNodes * Dimension
        self.ElementSize = self.LocalSize * 2

        # nodal forces
        self.qe = np.zeros(self.ElementSize)

        # transformation matrix
        self.S = self._get_transformation_matrix()

        self._print_element_information()

    def _print_element_information(self):
        msg = str(self.domain_size) + " Co-Rotational Beam Element\n"
        msg += "Element Size: " + str(self.ElementSize) + "\n"
        print(msg)

    def get_el_mass(self, i):
        """
            element mass matrix derivation from Klaus Bernd Sautter's master thesis
        """

    def _get_transformation_matrix(self):
        L = self.Li
        S = np.zeros([self.ElementSize, self.LocalSize])

        S[0, 3] = -1.00
        S[1, 5] = 2.00 / L
        S[2, 4] = -2.00 / L
        S[3, 0] = -1.00
        S[4, 1] = -1.00
        S[4, 4] = 1.00
        S[5, 2] = -1.00
        S[5, 5] = 1.00
        S[6, 3] = 1.00
        S[7, 5] = -2.00 / L
        S[8, 4] = 2.00 / L
        S[9, 0] = 1.00
        S[10, 1] = 1.00
        S[10, 4] = 1.00
        S[11, 2] = 1.00
        S[11, 5] = 1.00

        return S

    def get_el_stiffness(self, i):
        """
            element stiffness matrix derivation from Klaus Bernd Sautter's master thesis
        """
        pass

    def _get_local_stiffness_matrix_material(self, i):
        """
            elastic part of the total stiffness matrix
        """

        # shear coefficients
        Psi_y = 1 / (1 + 12 * self.E * self.Iy[i] / (self.Li ** 2 * self.G * self.Asz[i]))
        Psi_z = 1 / (1 + 12 * self.E * self.Iz[i] / (self.Li ** 2 * self.G * self.Asy[i]))

        ke_const = np.zeros([self.ElementSize, self.ElementSize])

        self.Li3 = self.Li * self.Li * self.Li
        self.Li2 = self.Li * self.Li

        ke_const[0, 0] = self.E * self.A[i] / self.Li
        ke_const[6, 0] = -1.0 * ke_const[0, 0]
        ke_const[0, 6] = ke_const[6, 0]
        ke_const[6, 6] = ke_const[0, 0]

        ke_const[1, 1] = 12.0 * self.E * self.Iz[i] * Psi_z / self.Li3
        ke_const[1, 7] = -1.0 * ke_const[1, 1]
        ke_const[1, 5] = 6.0 * self.E * self.Iz[i] * Psi_z / self.Li2
        ke_const[1, 11] = ke_const[1, 5]

        ke_const[2, 2] = 12.0 * self.E * self.Iy[i] * Psi_y / self.Li3
        ke_const[2, 8] = -1.0 * ke_const[2, 2]
        ke_const[2, 4] = -6.0 * self.E * self.Iy[i] * Psi_y / self.Li2
        ke_const[2, 10] = ke_const[2, 4]

        ke_const[4, 2] = ke_const[2, 4]
        ke_const[5, 1] = ke_const[1, 5]
        ke_const[3, 3] = self.G * self.It[i] / self.Li
        ke_const[4, 4] = self.E * self.Iy[i] * (3.0 * Psi_y + 1.0) / self.Li
        ke_const[5, 5] = self.E * self.Iz[i] * (3.0 * Psi_z + 1.0) / self.Li
        ke_const[4, 8] = -1.0 * ke_const[4, 2]
        ke_const[5, 7] = -1.0 * ke_const[5, 1]
        ke_const[3, 9] = -1.0 * ke_const[3, 3]
        ke_const[4, 10] = self.E * self.Iy[i] * (3.0 * Psi_y - 1.0) / self.Li
        ke_const[5, 11] = self.E * self.Iz[i] * (3.0 * Psi_z - 1.0) / self.Li

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

    def _get_local_stiffness_matrix_geometry(self, i):
        """
            geometric part of the total stiffness matrix
        """

        N = self.qe[6]
        Mt = self.qe[9]
        my_A = self.qe[4]
        mz_A = self.qe[5]
        my_B = self.qe[10]
        mz_B = self.qe[11]

        L = self.Li
        Qy = -1.00 * (mz_A + mz_B) / L
        Qz = (my_A + my_B) / L

        kg_const = np.zeros([self.ElementSize, self.ElementSize])

        kg_const[0, 1] = -Qy / L
        kg_const[0, 2] = -Qz / L
        kg_const[0, 7] = -1.0 * kg_const[0, 1]
        kg_const[0, 8] = -1.0 * kg_const[0, 2]

        kg_const[1, 0] = kg_const[0, 1]

        kg_const[1, 1] = 1.2 * N / L

        kg_const[1, 3] = my_A / L
        kg_const[1, 4] = Mt / L

        kg_const[1, 5] = N / 10.0

        kg_const[1, 6] = kg_const[0, 7]
        kg_const[1, 7] = -1.00 * kg_const[1, 1]
        kg_const[1, 9] = my_B / L
        kg_const[1, 10] = -1.00 * kg_const[1, 4]
        kg_const[1, 11] = kg_const[1, 5]

        kg_const[2, 0] = kg_const[0, 2]
        kg_const[2, 2] = kg_const[1, 1]
        kg_const[2, 3] = mz_A / L
        kg_const[2, 4] = -1.00 * kg_const[1, 5]
        kg_const[2, 5] = kg_const[1, 4]
        kg_const[2, 6] = kg_const[0, 8]
        kg_const[2, 8] = kg_const[1, 7]
        kg_const[2, 9] = mz_B / L
        kg_const[2, 10] = kg_const[2, 4]
        kg_const[2, 11] = kg_const[1, 10]

        for i in range(3):
            kg_const[3, i] = kg_const[i, 3]

        kg_const[3, 4] = (-mz_A / 3.00) + (mz_B / 6.00)
        kg_const[3, 5] = (my_A / 3.00) - (my_B / 6.00)
        kg_const[3, 7] = -my_A / L
        kg_const[3, 8] = -mz_A / L
        kg_const[3, 10] = L * Qy / 6.00
        kg_const[3, 11] = L * Qz / 6.00

        for i in range(4):
            kg_const[4, i] = kg_const[i, 4]

        kg_const[4, 4] = 2.00 * L * N / 15.00
        kg_const[4, 7] = -Mt / L
        kg_const[4, 8] = N / 10.00
        kg_const[4, 9] = kg_const[3, 10]
        kg_const[4, 10] = -L * N / 30.00
        kg_const[4, 11] = Mt / 2.00

        for i in range(5):
            kg_const[5, i] = kg_const[i, 5]

        kg_const[5, 5] = kg_const[4, 4]
        kg_const[5, 7] = -N / 10.0
        kg_const[5, 8] = -Mt / L
        kg_const[5, 9] = kg_const[3, 11]
        kg_const[5, 10] = -1.00 * kg_const[4, 11]
        kg_const[5, 11] = kg_const[4, 10]

        for i in range(6):
            kg_const[6, i] = kg_const[i, 6]

        kg_const[6, 7] = kg_const[0, 1]
        kg_const[6, 8] = kg_const[0, 2]

        for i in range(7):
            kg_const[7, i] = kg_const[i, 7]

        kg_const[7, 7] = kg_const[1, 1]
        kg_const[7, 9] = -1.00 * kg_const[1, 9]
        kg_const[7, 10] = kg_const[4, 1]
        kg_const[7, 11] = kg_const[2, 4]

        for i in range(8):
            kg_const[8, i] = kg_const[i, 8]

        kg_const[8, 8] = kg_const[1, 1]
        kg_const[8, 9] = -1.00 * kg_const[2, 9]
        kg_const[8, 10] = kg_const[1, 5]
        kg_const[8, 11] = kg_const[1, 4]

        for i in range(9):
            kg_const[9, i] = kg_const[i, 9]

        kg_const[9, 10] = (mz_A / 6.00) - (mz_B / 3.00)
        kg_const[9, 11] = (-my_A / 6.00) + (my_B / 3.00)

        for i in range(10):
            kg_const[10, i] = kg_const[i, 10]

        kg_const[10, 10] = kg_const[4, 4]

        for i in range(11):
            kg_const[11, i] = kg_const[i, 11]

        kg_const[11, 11] = kg_const[4, 4]

        return kg_const
