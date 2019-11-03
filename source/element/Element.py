import numpy as np


class Element(object):
    def __init__(self, material_params, element_params, nodal_coords, index, domain_size):
        self.material_params = material_params
        self.domain_size = domain_size

        self.isNonlinear = material_params['is_nonlinear']

        # nodal index - defined along the x axis
        self.index = index

        # material properties
        self.E = self.material_params['e']
        self.rho = self.material_params['rho']
        self.nu = self.material_params['nu']

        # area
        self.A = element_params['a']
        # effective area of shear
        self.Asy = element_params['asy']
        self.Asz = element_params['asz']

        # second moment of inertia
        self.Iy = element_params['iy']
        self.Iz = element_params['iz']
        # torsion constant J
        self.It = element_params['it']

        # element properties
        self.NumberOfNodes = 2
        self.Dimension = 3
        self.LocalSize = self.NumberOfNodes * self.Dimension
        self.ElementSize = self.LocalSize * 2

        # element geometry
        # element geometry Node A, Node B
        self.ReferenceCoords = nodal_coords.reshape(self.LocalSize)
        # reference length of one element
        self.L = self._calculate_reference_length()

        # nonlinear elements needs the nodal forces and deformations for the geometric stiffness calculation
        if self.isNonlinear:
            # nodal forces
            self.nodal_force_local = np.zeros(self.ElementSize)
            self.nodal_force_global = np.zeros(self.ElementSize)

            # [A_disp_x, A_disp_y, rot ..., B_disp_x, B_disp_y, ... rot ..]
            # placeholder for one time step deformation to calculate the increment
            self.current_deformation = np.zeros(self.ElementSize)
            self.previous_deformation = np.zeros(self.ElementSize)

    def _print_element_information(self):
        if self.isNonlinear:
            msg = "Nonlinear "
        else:
            msg = "Linear "
        msg += str(self.domain_size) + " Base Class Element " + str(self.index) + "\n"
        print(msg)

    def evaluate_torsional_inertia(self):
        # polar moment of inertia
        # assuming equivalency with circle
        self.Ip = self.Iy + self.Iz

    def evaluate_relative_importance_of_shear(self):
        self.G = self.E / 2 / (1 + self.nu)
        # relative importance of the shear deformation to the bending one
        self.Py = 12 * self.E * self.Iz / (self.G * self.Asy * self.L ** 2)
        self.Pz = 12 * self.E * self.Iy / (self.G * self.Asz * self.L ** 2)

    def get_element_stiffness_matrix(self):
        ke = self._get_element_stiffness_matrix_material()

        if self.isNonlinear:
            ke += self._get_element_stiffness_matrix_geometry()

        return ke

    def _get_element_stiffness_matrix_material(self):
        pass

    def _get_element_stiffness_matrix_geometry(self):
        pass

    def get_element_mass_matrix(self):
        pass

    def _calculate_reference_length(self):
        dx = self.ReferenceCoords[0] - self.ReferenceCoords[3]
        dy = self.ReferenceCoords[1] - self.ReferenceCoords[4]
        dz = self.ReferenceCoords[2] - self.ReferenceCoords[5]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        return length

    def _calculate_current_length(self):
        du = self.current_deformation[6] - self.current_deformation[0]
        dv = self.current_deformation[7] - self.current_deformation[1]
        dw = self.current_deformation[8] - self.current_deformation[2]

        dx = self.ReferenceCoords[3] - self.ReferenceCoords[0]
        dy = self.ReferenceCoords[4] - self.ReferenceCoords[1]
        dz = self.ReferenceCoords[5] - self.ReferenceCoords[2]

        length = np.sqrt((du + dx) * (du + dx) + (dv + dy) * (dv + dy) + (dw + dz) * (dw + dz))
        return length

    def _get_current_nodal_position(self):
        # element current nodal positions
        CurrentCoords = np.zeros(self.LocalSize)

        for i in range(self.NumberOfNodes):
            k = i * self.Dimension
            j = i * self.LocalSize
            CurrentCoords[k] = self.ReferenceCoords[k] + self.current_deformation[j]
            CurrentCoords[k + 1] = self.ReferenceCoords[k + 1] + self.current_deformation[j + 1]
            CurrentCoords[k + 2] = self.ReferenceCoords[k + 2] + self.current_deformation[j + 2]

        return CurrentCoords

    def _assign_new_deformation(self, new_deformation):
        self.previous_deformation = self.current_deformation
        self.current_deformation = new_deformation

    def _update_increment_deformation(self):
        """
         This function updates incremental deformation w.r.t. to current and previous deformations
        """
        increment_deformation = self.current_deformation - self.previous_deformation
        return increment_deformation
