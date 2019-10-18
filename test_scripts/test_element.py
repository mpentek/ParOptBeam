from source.element.CRBeamElement import CRBeamElement

parameters = {'rho': 10.0, 'e': 100., 'nu': 0.1, 'zeta': 0.05, 'lx_i': 10.}

# # length of one element - assuming an equidistant grid


def test():
    element = CRBeamElement(parameters, '3D')
    element.A = [5., 5.]
    element.Asy = [2., 2.]
    element.Asz = [1., 1.]
    element.Iy = [10., 10.]
    element.Iz = [20., 20.]
    element.It = [20., 30.]

    element.Py = [12 * element.E * a / (
            element.G * b * element.Li ** 2) for a, b in
                       zip(element.Iz, element.Asy)]
    element.Pz = [12 * element.E * a / (
            element.G * b * element.Li ** 2) for a, b in
                       zip(element.Iy, element.Asz)]

    element.get_el_stiffness(0)


if __name__ == '__main__':
    test()
