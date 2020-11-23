import numpy as np

from source.element.beam_element import BeamElement


class TimoshenkoBeamElement(BeamElement):
    def __init__(self, material_params, element_params, nodal_coords, index, domain_size):

        # TODO: we should not aim to handle material nonlinearity - or it seems more likely to be a naming problem
        if material_params['is_nonlinear']:
            err_msg = "Nonlinear TimoshenkoBeamElement is not yet implemented"
            raise Exception(err_msg)
        super().__init__(material_params, element_params, nodal_coords, index, domain_size)

        self.evaluate_torsional_inertia()
        self.evaluate_relative_importance_of_shear()

        self._print_element_information()

    def _print_element_information(self):
        msg = str(self.domain_size) + " Timoshenko Beam Element " + str(self.index) + "\n"
        msg += "Initial coordinates: \n"
        msg += str(self.ReferenceCoords[:3]) + "\n"
        msg += str(self.ReferenceCoords[3:]) + "\n"
        msg += "A: " + str(self.A) + "\n"
        msg += "Asy: " + str(self.Asy) + "\n"
        msg += "Asz: " + str(self.Asz) + "\n"
        msg += "Iy: " + str(self.Iy) + "\n"
        msg += "Iz: " + str(self.Iz) + "\n"
        msg += "Pz: " + str(self.Pz) + "\n"
        msg += "Py: " + str(self.Py) + "\n"
        print(msg)

    def get_element_mass_matrix(self):
        """
        Getting the consistant mass matrix based on analytical integration

        USING the consistent mass formulation

        mass values for one level
        VERSION 3: from Appendix A - Straight Beam Element Matrices - page 228
        https://link.springer.com/content/pdf/bbm%3A978-3-319-56493-7%2F1.pdf
        """

        m_const = self.rho * self.A * self.L

        #
        # mass values for one level
        # define component-wise to have enable better control for various optimization parameters

        # axial inertia - along axis x - here marked as x
        m_x = m_const / 6.0
        m_x_11 = 2.
        m_x_12 = 1.
        m_el_x = m_x * np.array([[m_x_11, m_x_12],
                                 [m_x_12, m_x_11]])

        if self.domain_size == '3D':
            # torsion inertia - around axis x - here marked as alpha - a
            m_a = m_const * self.Ip / self.A / 6.0
            m_a_11 = 2
            m_a_12 = 1
            m_el_a = m_a * np.array([[m_a_11, m_a_12],
                                     [m_a_12, m_a_11]])

        # bending - inertia along axis y, rotations around axis z - here marked as gamma - g
        # translation
        Py = self.Py
        m_yg = m_const / 210 / (1 + Py) ** 2
        #
        m_yg_11 = 70 * Py ** 2 + 147 * Py + 78
        m_yg_12 = (35 * Py ** 2 + 77 * Py + 44) * self.L / 4
        m_yg_13 = 35 * Py ** 2 + 63 * Py + 27
        m_yg_14 = -(35 * Py ** 2 + 63 * Py + 26) * self.L / 4
        #
        m_yg_22 = (7 * Py ** 2 + 14 * Py + 8) * self.L ** 2 / 4
        m_yg_23 = - m_yg_14
        m_yg_24 = -(7 * Py ** 2 + 14 * Py + 6) * self.L ** 2 / 4
        #
        m_yg_33 = m_yg_11
        m_yg_34 = -m_yg_12
        #
        m_yg_44 = m_yg_22
        #
        m_el_yg_trans = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                         [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                         [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                         [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])
        # rotation
        m_yg = self.rho * self.Iz / 30 / (1 + Py) ** 2 / self.L
        #
        m_yg_11 = 36
        m_yg_12 = -(15 * Py - 3) * self.L
        m_yg_13 = -m_yg_11
        m_yg_14 = m_yg_12
        #
        m_yg_22 = (10 * Py ** 2 + 5 * Py + 4) * self.L ** 2
        m_yg_23 = - m_yg_12
        m_yg_24 = (5 * Py ** 2 - 5 * Py - 1) * self.L ** 2
        #
        m_yg_33 = m_yg_11
        m_yg_34 = - m_yg_12
        #
        m_yg_44 = m_yg_22
        #
        m_el_yg_rot = m_yg * np.array([[m_yg_11, m_yg_12, m_yg_13, m_yg_14],
                                       [m_yg_12, m_yg_22, m_yg_23, m_yg_24],
                                       [m_yg_13, m_yg_23, m_yg_33, m_yg_34],
                                       [m_yg_14, m_yg_24, m_yg_34, m_yg_44]])

        # sum up translation and rotation
        m_el_yg = m_el_yg_trans + m_el_yg_rot

        if self.domain_size == '3D':
            # bending - inertia along axis z, rotations around axis y - here marked as beta - b
            # translation
            Pz = self.Pz
            m_zb = m_const / 210 / (1 + Pz) ** 2
            #
            m_zb_11 = 70 * Pz ** 2 + 147 * Pz + 78
            m_zb_12 = -(35 * Pz ** 2 + 77 * Pz + 44) * self.L / 4.
            m_zb_13 = 35 * Pz ** 2 + 63 * Pz + 27
            m_zb_14 = (35 * Pz ** 2 + 63 * Pz + 26) * self.L / 4.
            #
            m_zb_22 = (7 * Pz ** 2 + 14 * Pz + 8) * self.L ** 2 / 4.
            m_zb_23 = -m_zb_14
            m_zb_24 = -(7 * Pz ** 2 + 14 * Pz + 6) * self.L ** 2 / 4.
            #
            m_zb_33 = m_zb_11
            m_zb_34 = - m_zb_12
            #
            m_zb_44 = m_zb_22
            #
            m_el_zb_trans = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                             [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                             [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                             [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])
            # rotation
            m_zb = self.rho * self.Iy / 30. / (1 + Pz) ** 2 / self.L

            m_zb_11 = 36.
            m_zb_12 = (15. * Pz - 3) * self.L
            m_zb_13 = -m_zb_11
            m_zb_14 = m_zb_12

            m_zb_22 = (10 * Pz ** 2 + 5 * Pz + 4) * self.L ** 2
            m_zb_23 = -m_zb_12
            m_zb_24 = (5 * Pz ** 2 - 5 * Pz - 1) * self.L ** 2

            m_zb_33 = m_zb_11
            m_zb_34 = -m_zb_12

            m_zb_44 = m_zb_22

            m_el_zb_rot = m_zb * np.array([[m_zb_11, m_zb_12, m_zb_13, m_zb_14],
                                           [m_zb_12, m_zb_22, m_zb_23, m_zb_24],
                                           [m_zb_13, m_zb_23, m_zb_33, m_zb_34],
                                           [m_zb_14, m_zb_24, m_zb_34, m_zb_44]])

            # sum up translation and rotation
            m_el_zb = m_el_zb_trans + m_el_zb_rot

        # assemble all components
        if self.domain_size == '3D':
            m_el = np.array([[m_el_x[0][0], 0., 0., 0., 0., 0., m_el_x[0][1], 0., 0., 0., 0., 0.],
                             [0., m_el_yg[0][0], 0., 0., 0., m_el_yg[0][1], 0., m_el_yg[0][2], 0., 0., 0.,
                              m_el_yg[0][3]],
                             [0., 0., m_el_zb[0][0], 0., m_el_zb[0][1], 0., 0., 0., m_el_zb[0][2], 0., m_el_zb[0][3],
                              0.],
                             [0., 0., 0., m_el_a[0][0], 0., 0., 0., 0., 0., m_el_a[0][1], 0., 0.],
                             [0., 0., m_el_zb[0][1], 0., m_el_zb[1][1], 0., 0., 0., m_el_zb[1][2], 0., m_el_zb[1][3],
                              0.],
                             [0., m_el_yg[0][1], 0., 0., 0., m_el_yg[1][1], 0., m_el_yg[1][2], 0., 0., 0.,
                              m_el_yg[1][3]],

                             [m_el_x[1][0], 0., 0., 0., 0., 0., m_el_x[1][1], 0., 0., 0., 0., 0.],
                             [0., m_el_yg[0][2], 0., 0., 0., m_el_yg[1][2], 0., m_el_yg[2][2], 0., 0., 0.,
                              m_el_yg[2][3]],
                             [0., 0., m_el_zb[0][2], 0., m_el_zb[1][2], 0., 0., 0., m_el_zb[2][2], 0., m_el_zb[2][3],
                              0.],
                             [0., 0., 0., m_el_a[1][0], 0., 0., 0., 0., 0., m_el_a[1][1], 0., 0.],
                             [0., 0., m_el_zb[0][3], 0., m_el_zb[1][3], 0., 0., 0., m_el_zb[2][3], 0., m_el_zb[3][3],
                              0.],
                             [0., m_el_yg[0][3], 0., 0., 0., m_el_yg[1][3], 0., m_el_yg[2][3], 0., 0., 0.,
                              m_el_yg[3][3]]])

        elif self.domain_size == '2D':
            m_el = np.array([[m_el_x[0][0], 0., 0., m_el_x[0][1], 0., 0.],
                             [0., m_el_yg[0][0], m_el_yg[0][1], 0., m_el_yg[0][2], m_el_yg[0][3]],
                             [0., m_el_yg[0][1], m_el_yg[1][1], 0., m_el_yg[1][2], m_el_yg[1][3]],

                             [m_el_x[1][0], 0., 0., m_el_x[1][1], 0., 0.],
                             [0., m_el_yg[0][2], m_el_yg[1][2], 0., m_el_yg[2][2], m_el_yg[2][3]],
                             [0., m_el_yg[0][3], m_el_yg[1][3], 0., m_el_yg[2][3], m_el_yg[3][3]]])

        return m_el

    # TODO implement geometry contribution of the timoshenko beam
    def _get_element_stiffness_matrix_geometry(self):
        pass

    def _get_element_stiffness_matrix_material(self):
        """
        stiffness values for one level
        VERSION 2

        NOTE: from http://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf
        seems to be a typo in 1-105 and 1-106 as a division with l**3 instead of l**3 should take place
        implemented mass matrices similar to the stiffness one
        """

        # axial stiffness - along axis x - here marked as x
        k_x = self.E * self.A / self.L
        k_x_11 = 1.0
        k_x_12 = -1.0
        k_el_x = k_x * np.array([[k_x_11, k_x_12],
                                 [k_x_12, k_x_11]])

        if self.domain_size == '3D':
            # unncecessary if k_ya is used
            # torsion stiffness - around axis x - here marked as alpha - a
            k_a = self.G * \
                  self.It / self.L
            k_a_11 = 1.0
            k_a_12 = -1.0
            k_el_a = k_a * np.array([[k_a_11, k_a_12],
                                     [k_a_12, k_a_11]])
        
        if self.domain_size == '3D':
            # bending - displacement along y, rotations around axis x due to exzenticity in z direchtion ez
            k_aa = self.G * self.It / self.L 
            k_ya = self.YT / self.L
                    
            k_ya_11 = 0. # from k_el_yg
            k_ya_12 = k_ya
            k_ya_13 = - k_ya_11
            k_ya_14 = - k_ya_12
            k_ya_22 = k_aa
            k_ya_23 = k_ya_14
            k_ya_24 = - k_ya_22 
            k_ya_33 = k_ya_11
            k_ya_34 = k_ya_12
            k_ya_44 = k_ya_22
            k_el_ya = np.array([[k_ya_11, k_ya_12, k_ya_13, k_ya_14],
                                [k_ya_12, k_ya_22, k_ya_23, k_ya_24],
                                [k_ya_13, k_ya_23, k_ya_33, k_ya_34],
                                [k_ya_14, k_ya_24, k_ya_34, k_ya_44]])

        # bending - displacement along axis y, rotations around axis z - here marked as gamma - g
        beta_yg = self.Py
        k_yg = self.E * self.Iz / \
               (1 + beta_yg) / self.L ** 3
        #
        k_yg_11 = 12.
        k_yg_12 = 6. * self.L
        k_yg_13 = -k_yg_11
        k_yg_14 = k_yg_12
        #
        k_yg_22 = (4. + beta_yg) * self.L ** 2
        k_yg_23 = -k_yg_12
        k_yg_24 = (2 - beta_yg) * self.L ** 2
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
            beta_zb = self.Pz
            k_zb = self.E * self.Iy / \
                   (1 + beta_zb) / self.L ** 3
            #
            k_zb_11 = 12.
            k_zb_12 = -6. * self.L
            k_zb_13 = -12.
            k_zb_14 = k_zb_12
            #
            k_zb_22 = (4. + beta_zb) * self.L ** 2
            k_zb_23 = -k_zb_12
            k_zb_24 = (2 - beta_zb) * self.L ** 2
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
                             [0., k_el_yg[0][0]+k_el_ya[0][0], 0., k_el_ya[0][1], 0., k_el_yg[0][1], 0., 
                              k_el_yg[0][2]+k_el_ya[0][2], 0., k_el_ya[0][3], 0., k_el_yg[0][3]],
                             [0., 0., k_el_zb[0][0], 0., k_el_zb[0][1], 0., 0., 0., k_el_zb[0][2], 0., k_el_zb[0][3],
                              0.],
                             [0., k_el_ya[1][0], 0., k_el_ya[1][1], 0., 0., 0., k_el_ya[1][2], 0., k_el_ya[1][3], 0., 0.],
                             [0., 0., k_el_zb[0][1], 0., k_el_zb[1][1], 0., 0., 0., k_el_zb[1][2], 0., k_el_zb[1][3],
                              0.],
                             [0., k_el_yg[0][1], 0., 0., 0., k_el_yg[1][1], 0., k_el_yg[1][2], 0., 0., 0.,
                              k_el_yg[1][3]],

                             [k_el_x[1][0], 0., 0., 0., 0., 0., k_el_x[1][1], 0., 0., 0., 0., 0.],
                             [0., k_el_yg[0][2]+k_el_ya[2][0], 0., k_el_ya[2][1], 0., k_el_yg[1][2], 0., 
                              k_el_yg[2][2]+k_el_ya[2][2], 0., k_el_ya[2][3], 0., k_el_yg[2][3]],
                             [0., 0., k_el_zb[0][2],  k_el_ya[0][0], k_el_zb[1][2], 0., 0., 0., k_el_zb[2][2],  k_el_ya[0][1], k_el_zb[2][3],
                              0.],
                             [0., k_el_ya[3][0], 0., k_el_ya[3][1], 0., 0., 0., k_el_ya[3][2],  0., k_el_ya[3][3], 0., 0.],
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

    def _compute_stiffnes_matrix_material(self):
        # Kinematic stuff
        S = np.array([[1., 0., 0., 0., 0. ,0., -1 , 0., 0., 0., 0., 0.],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     []])       