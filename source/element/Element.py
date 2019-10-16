class Element(object):
    def __init__(self, parameters, domain_size):
        self.parameters = parameters
        self.domain_size = domain_size

    def get_el_mass(self, i):
        pass

    def get_el_stiffness(self, i):
        pass

    def _print_element_information(self):
        pass
