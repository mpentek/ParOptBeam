import numpy as np

from source.element.TimoshenkoBeamElement import TimoshenkoBeamElement


class BernoulliBeamElement(TimoshenkoBeamElement):
    def __init__(self, material_params, element_params, nodal_coords, index, domain_size):
        super().__init__(material_params, element_params, nodal_coords, index, domain_size)

    def _print_element_information(self):
        print(str(self.domain_size), " Bernoulli Beam Element")

    def evaluate_relative_importance_of_shear(self):
        # relative importance of the shear deformation to the bending one
        self.Py = 0.
        self.Pz = 0.
