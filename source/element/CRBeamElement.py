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
        # shear coefficients
        phi_a_y = 1 / (1 + 12 * self.E * self.Iy / (self.Li ** 2 * self.G * self.Asz))
        phi_a_z = 1 / (1 + 12 * self.E * self.Iz / (self.Li ** 2 * self.G * self.Asy))

        # constant contribution to Ke
        # (eq. 4.158)
        kc_11 = np.array([[self.E * self.A / self.Li, 0., 0.],
                          [0., 12 * self.E * self.Iz * phi_a_z / (self.Li ** 3), 0.],
                          [0., 12 * self.E * self.Iy * phi_a_y / (self.Li ** 3), 0.]])

        kc_33 = kc_11
        kc_13 = -kc_11
        kc_31 = -kc_11

        # (eq. 4.159)
        kc_22 = np.array([[self.G * self.It / self.Li, 0., 0.],
                          [0., (3 * phi_a_y + 1) * self.E * self.Iy / self.Li, 0.],
                          [0., 0., (3 * phi_a_z + 1) * self.E * self.Iz / self.Li]])

        kc_44 = kc_22

        # (eq. 4.160.)
        kc_24 = np.array([[-self.G * self.It / self.Li, 0., 0.],
                          [0., (3 * phi_a_y - 1) * self.E * self.Iy / self.Li, 0.],
                          [0., 0., (3 * phi_a_z - 1) * self.E * self.Iz / self.Li]])

        kc_42 = kc_24

        kc_12 = np.array([[0., 0., 0.],
                          [0., 0., 6 * phi_a_z * self.E * self.Iy / (self.Li ** 2)],
                          [0., 6 * phi_a_y * self.E * self.Iz / (self.Li ** 2), 0.]])

        kc_14 = kc_12
        kc_23 = -np.transpose(kc_12)
        kc_43 = kc_23
        kc_21 = np.transpose(kc_12)
        kc_41 = kc_21
        kc_32 = -kc_12
        kc_34 = -kc_12

        Ke_const = np.array([[kc_11, kc_12, kc_13, kc_14],
                             [kc_21, kc_22, kc_23, kc_24],
                             [kc_31, kc_32, kc_33, kc_34],
                             [kc_41, kc_42, kc_43, kc_44]])

        print(Ke_const)

        # geometric contribution to Ke
        # kg_11 = np.array([[0., -Qy / l., 0.],
        #                   [0., 0., 6 * phi_a_z * self.E * self.Iy / (self.Li ** 2)],
        #                   [0., 6 * phi_a_y * self.E * self.Iz / (self.Li ** 2), 0.]])


        return Ke_const
