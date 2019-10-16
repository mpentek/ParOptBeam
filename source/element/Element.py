
class Element(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.domain_size = parameters["domain_size"]

    def get_el_stiffness(self, i):
        pass
