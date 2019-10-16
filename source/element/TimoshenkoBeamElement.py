import numpy as np

from source.element.Element import Element


class TimoshenkoBeamElement(Element):
    def __init__(self, parameters, domain_size):
        super().__init__(parameters, domain_size)

    def _print_element_information(self):
        print(str(self.domain_size), "D Timoshenko Beam Element")

    def get_el_stiffness(self, i):
        """
        stiffness values for one level
        VERSION 2

        NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
        seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
        implemented mass matrices similar to the stiffness one
        """

        # axial stiffness - along axis x - here marked as x
        k_x = self.parameters['e'] * \
              self.parameters['a'][i] / self.parameters['lx_i']
        k_x_11 = 1.0
        k_x_12 = -1.0
        k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                 [k_x_12, k_x_11]])

        if self.domain_size == '3D':
            # torsion stiffness - around axis x - here marked as alpha - a
            k_a = self.parameters['g'] * \
                  self.parameters['it'][i] / self.parameters['lx_i']
            k_a_11 = 1.0
            k_a_12 = -1.0
            k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                     [k_a_12, k_a_11]])

        # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
        beta_yg = self.parameters['py'][i]
        k_yg = self.parameters['e'] * self.parameters['iz'][i] / \
               (1 + beta_yg) / self.parameters['lx_i'] ** 3
        #
        k_yg_11 = 12.
        k_yg_12 = 6. * self.parameters['lx_i']
        k_yg_13 = -k_yg_11
        k_yg_14 = k_yg_12
        #
        k_yg_22 = (4. + beta_yg) * self.parameters['lx_i'] ** 2
        k_yg_23 = -k_yg_12
        k_yg_24 = (2 - beta_yg) * self.parameters['lx_i'] ** 2
        #
        k_yg_33 = k_yg_11
        k_yg_34 = -k_yg_12
        #
        k_yg_44 = k_yg_22
        #
        k_el_yg = k_yg * np.array([[k_yg_11, k_yg_12, k_yg_13, k_yg_14],
                                   [k_yg_12, k_yg_22, k_yg_23, k_yg_24],
                                   [k_yg_13, k_yg_23, k_yg_33, k_yg_34],
                                   [k_yg_14, k_yg_24, k_yg_34, k_yg_44]])

        if self.domain_size == '3D':
            # bending - displacement along axis z, rotations around axis y - here marked as beta - b
            beta_zb = self.parameters['pz'][i]
            k_zb = self.parameters['e'] * self.parameters['iy'][i] / \
                   (1 + beta_zb) / self.parameters['lx_i'] ** 3
            #
            k_zb_11 = 12.
            k_zb_12 = -6. * self.parameters['lx_i']
            k_zb_13 = -12.
            k_zb_14 = k_zb_12
            #
            k_zb_22 = (4. + beta_zb) * self.parameters['lx_i'] ** 2
            k_zb_23 = -k_zb_12
            k_zb_24 = (2 - beta_zb) * self.parameters['lx_i'] ** 2
            #
            k_zb_33 = k_zb_11
            k_zb_34 = - k_zb_12
            #
            k_zb_44 = k_zb_22
            #
            k_el_zb = k_zb * np.array([[k_zb_11, k_zb_12, k_zb_13, k_zb_14],
                                       [k_zb_12, k_zb_22, k_zb_23, k_zb_24],
                                       [k_zb_13, k_zb_23, k_zb_33, k_zb_34],
                                       [k_zb_14, k_zb_24, k_zb_34, k_zb_44]])

        if self.domain_size == '3D':
            # assemble all components
            k_el = np.array([[k_el_x[0][0], 0., 0., 0., 0., 0., k_el_x[0][1], 0., 0., 0., 0., 0.],
                             [0., k_el_yg[0][0], 0., 0., 0., k_el_yg[0][1], 0., k_el_yg[0][2], 0., 0., 0.,
                              k_el_yg[0][3]],
                             [0., 0., k_el_zb[0][0], 0., k_el_zb[0][1], 0., 0., 0., k_el_zb[0][2], 0., k_el_zb[0][3],
                              0.],
                             [0., 0., 0., k_el_a[0][0], 0., 0., 0., 0., 0., k_el_a[0][1], 0., 0.],
                             [0., 0., k_el_zb[0][1], 0., k_el_zb[1][1], 0., 0., 0., k_el_zb[1][2], 0., k_el_zb[1][3],
                              0.],
                             [0., k_el_yg[0][1], 0., 0., 0., k_el_yg[1][1], 0., k_el_yg[1][2], 0., 0., 0.,
                              k_el_yg[1][3]],

                             [k_el_x[1][0], 0., 0., 0., 0., 0., k_el_x[1][1], 0., 0., 0., 0., 0.],
                             [0., k_el_yg[0][2], 0., 0., 0., k_el_yg[1][2], 0., k_el_yg[2][2], 0., 0., 0.,
                              k_el_yg[2][3]],
                             [0., 0., k_el_zb[0][2], 0., k_el_zb[1][2], 0., 0., 0., k_el_zb[2][2], 0., k_el_zb[2][3],
                              0.],
                             [0., 0., 0., k_el_a[1][0], 0., 0., 0., 0., 0., k_el_a[1][1], 0., 0.],
                             [0., 0., k_el_zb[0][3], 0., k_el_zb[1][3], 0., 0., 0., k_el_zb[2][3], 0., k_el_zb[3][3],
                              0.],
                             [0., k_el_yg[0][3], 0., 0., 0., k_el_yg[1][3], 0., k_el_yg[2][3], 0., 0., 0.,
                              k_el_yg[3][3]]])

        elif self.domain_size == '2D':
            k_el = np.array([[k_el_x[0][0], 0., 0., k_el_x[0][1], 0., 0.],
                             [0., k_el_yg[0][0], k_el_yg[0][1], 0., k_el_yg[0][2], k_el_yg[0][3]],
                             [0., k_el_yg[0][1], k_el_yg[1][1], 0., k_el_yg[1][2], k_el_yg[1][3]],

                             [k_el_x[1][0], 0., 0., k_el_x[1][1], 0., 0.],
                             [0., k_el_yg[0][2], k_el_yg[1][2], 0., k_el_yg[2][2], k_el_yg[2][3]],
                             [0., k_el_yg[0][3], k_el_yg[1][3], 0., k_el_yg[2][3], k_el_yg[3][3]]])

        return k_el
