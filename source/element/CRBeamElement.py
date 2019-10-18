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
        # torsion constant
        self.It = None
        # evaluating torsional inertia
        self.Ip = None

        self.Py = None
        self.Pz = None

        self._print_element_information()

    def _print_element_information(self):
        print(str(self.domain_size), "D Co-Rotational Beam Element")

    def get_el_mass(self, i):
        """
            element mass matrix derivation from Klaus Bernd Sautter's master thesis
        """

    def get_el_stiffness(self, i):
        """
            element stiffness matrix derivation from Klaus Bernd Sautter's master thesis
        """

        # placeholder for the stiffness matrix
        Ke_const = np.zeros([6, 6])
        Ke_geo = np.zeros([6, 6])
        Ke_tot = np.zeros([6, 6])

        # placeholder for partial entities
        kc = np.zeros([4, 4, 3, 3])

        # shear coefficients
        phi_a_y = 1 / (1 + 12 * self.E * self.Iy[i] / (self.Li ** 2 * self.G * self.Asz[i]))
        phi_a_z = 1 / (1 + 12 * self.E * self.Iz[i] / (self.Li ** 2 * self.G * self.Asy[i]))

        # constant contribution to Ke
        # (eq. 4.158)
        kc[0][0] = np.array([[self.E * self.A[i] / self.Li, 0., 0.],
                             [0., 12 * self.E * self.Iz[i] * phi_a_z / (self.Li ** 3), 0.],
                             [0., 12 * self.E * self.Iy[i] * phi_a_y / (self.Li ** 3), 0.]])

        kc[2][2] = kc[0][0]
        kc[0][2] = -kc[0][0]
        kc[2][0] = -kc[0][0]

        # (eq. 4.159)
        kc[1][1] = np.array([[self.G * self.It[i] / self.Li, 0., 0.],
                             [0., (3 * phi_a_y + 1) * self.E * self.Iy[i] / self.Li, 0.],
                             [0., 0., (3 * phi_a_z + 1) * self.E * self.Iz[i] / self.Li]])

        kc[3][3] = kc[1][1]

        # (eq. 4.160.)
        kc[1][3] = np.array([[-self.G * self.It[i] / self.Li, 0., 0.],
                             [0., (3 * phi_a_y - 1) * self.E * self.Iy[i] / self.Li, 0.],
                             [0., 0., (3 * phi_a_z - 1) * self.E * self.Iz[i] / self.Li]])

        kc[3][1] = kc[1][3]

        kc[0][1] = np.array([[0., 0., 0.],
                             [0., 0., 6 * phi_a_z * self.E * self.Iy[i] / (self.Li ** 2)],
                             [0., 6 * phi_a_y * self.E * self.Iz[i] / (self.Li ** 2), 0.]])

        kc[0][3] = kc[0][1]
        kc[1][2] = -np.transpose(kc[0][1])
        kc[3][2] = kc[1][2]
        kc[1][0] = np.transpose(kc[0][1])
        kc[3][0] = kc[1][0]
        kc[2][1] = -kc[0][1]
        kc[2][3] = -kc[0][1]

        for i in range(4):
            for j in range(4):
                Ke_const[i:i+3, j:j+3] += kc[i][j]
        print(Ke_const)

        # geometric contribution to Ke
        # kg_11 = np.array([[0., -Qy / l., 0.],
        #                   [0., 0., 6 * phi_a_z * self.E * self.Iy / (self.Li ** 2)],
        #                   [0., 6 * phi_a_y * self.E * self.Iz / (self.Li ** 2), 0.]])

        return Ke_const
