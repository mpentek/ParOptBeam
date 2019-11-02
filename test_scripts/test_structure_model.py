from source.model.structure_model import StraightBeam
import numpy as np

params = {
    "name": "CaarcBeamPrototypeOptimizable",
    "domain_size": "3D",
    "system_parameters": {
        "element_params": {
            "type": "CRBeam",
            "is_nonlinear": True
        },
        "material": {
            "density": 160.0,
            "youngs_modulus": 2.861e8,
            "poisson_ratio": 0.1,
            "damping_ratio": 0.005
        },
        "geometry": {
            "length_x": 180.0,
            "number_of_elements": 3,
            "defined_on_intervals": [{
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
                    "outrigger_stiffness": [0.0]}]
        }
    },
    "boundary_conditions": "fixed-free"
}


def test_structure_model():
    beam = StraightBeam(params)

