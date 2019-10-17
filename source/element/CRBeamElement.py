import numpy as np

from source.element.Element import Element


class CRBeamElement(Element):
    def __init__(self, parameters, domain_size):
        super().__init__(parameters, domain_size)

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
