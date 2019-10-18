from source.element.CRBeamElement import CRBeamElement

parameters = {'rho': 160.0,
              'e': 2.861e8,
              'nu': 0.1,
              'zeta': 0.005,
              'lx': 180,
              'n_el': 3, 'intervals': []}

system_params = {
    "defined_on_intervals": [
        {
            "interval_bounds": [0.0, 60.0],
            "length_y": [50.0],
            "length_z": [35.0],
            "area": [1750.0],
            "shear_area_y": [1460.0],
            "shear_area_z": [1460.0],
            "moment_of_inertia_y": [178646.0],
            "moment_of_inertia_z": [364583.0],
            "torsional_moment_of_inertia": [420175.0],
            "outrigger_mass": [0.0],
            "outrigger_stiffness": [0.0]},
        {
            "interval_bounds": [60.0, 120.0],
            "length_y": [45.0],
            "length_z": [30.0],
            "area": [1350.0],
            "shear_area_y": [1125.0],
            "shear_area_z": [1125.0],
            "moment_of_inertia_y": [101250.0],
            "moment_of_inertia_z": [227812.5],
            "torsional_moment_of_inertia": [238140.0],
            "outrigger_mass": [0.0],
            "outrigger_stiffness": [0.0]},
        {
            "interval_bounds": [120.0, "End"],
            "length_y": [40.0],
            "length_z": [25.0],
            "area": [1000.0],
            "shear_area_y": [833.0],
            "shear_area_z": [833.0],
            "moment_of_inertia_y": [52083.0],
            "moment_of_inertia_z": [133333.0],
            "torsional_moment_of_inertia": [122500.0],
        }
    ]
}

# # length of one element - assuming an equidistant grid
parameters['lx_i'] = parameters['lx'] / parameters['n_el']


def test():
    element = CRBeamElement(parameters, '3D')
    element.A = 1350
    element.Asy = 1460
    element.Asz = 1460
    element.Iy = 101250
    element.Iz = 1125
    element.It = 122500

    element.Py = 12 * element.E * element.Iz / (
            element.G * element.Asy * element.Li ** 2)
    element.Pz = 12 * element.E * element.Iy / (
            element.G * element.Asz * element.Li ** 2)

    element.get_el_stiffness(0)


if __name__ == '__main__':
    test()
