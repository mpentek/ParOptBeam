import numpy as np


class Element(object):
    def __init__(self, material_params, element_params, nodal_coords, domain_size):
        self.material_params = material_params
        self.domain_size = domain_size

        # material properties
        self.E = self.material_params['e']
        self.rho = self.material_params['rho']
        self.nu = self.material_params['nu']
        self.G = self.material_params['g'] = self.E / 2 / (1 + self.nu)

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

        # element geometry Node A, Node B
        self.ReferenceCoords = nodal_coords.reshape(self.LocalSize)
        # element current nodal positions
        self.CurrentCoords = self.ReferenceCoords
        # reference length of one element
        self.L = self._calculate_reference_length()

    def get_element_stiffness_matrix(self):
        pass

    def get_element_mass_matrix(self):
        pass

    def _print_element_information(self):
        pass

    def _calculate_reference_length(self):
        dx = self.ReferenceCoords[0] - self.ReferenceCoords[1]
        dy = self.ReferenceCoords[2] - self.ReferenceCoords[3]
        dz = self.ReferenceCoords[4] - self.ReferenceCoords[5]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        return length
