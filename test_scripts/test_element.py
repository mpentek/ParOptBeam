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
            "outrigger_mass": [0.0],
            "outrigger_stiffness": [0.0]
        }
    ]
}

# defined on intervals as piecewise continuous function on an interval starting from 0.0
for val in system_params["defined_on_intervals"]:
    parameters["intervals"].append({
        'bounds': val['interval_bounds'],
        # further quantities defined by polynomial coefficient as a function of running coord x
        'c_ly': val["length_y"],
        'c_lz': val["length_z"],
        'c_a': val["area"],
        'c_a_sy': val["shear_area_y"],
        'c_a_sz': val["shear_area_z"],
        'c_iy': val["moment_of_inertia_y"],
        'c_iz': val["moment_of_inertia_z"],
        'c_it': val["torsional_moment_of_inertia"],
        'c_m': val["outrigger_mass"],
        'c_k': val["outrigger_stiffness"]
    })

element = CRBeamElement(parameters, '3D')
