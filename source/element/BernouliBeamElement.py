import numpy as np

from source.element.TimoshenkoBeamElement import TimoshenkoBeamElement


class BernoulliBeamElement(TimoshenkoBeamElement):
    def __init__(self, material_params, element_params, nodal_coords, index, domain_size):
        if material_params['is_nonlinear']:
            err_msg = "Nonlinear BernoulliBeamElement is not yet implemented"
            raise Exception(err_msg)
        super().__init__(material_params, element_params, nodal_coords, index, domain_size)

        self.evaluate_torsional_inertia()
        self.evaluate_relative_importance_of_shear()

    def _print_element_information(self):
        msg = str(self.domain_size) + " Bernoulli Beam Element " + str(self.index) + "\n"
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

    def evaluate_relative_importance_of_shear(self):
        self.G = self.E / 2 / (1 + self.nu)
        # relative importance of the shear deformation to the bending one
        self.Py = 0.
        self.Pz = 0.
