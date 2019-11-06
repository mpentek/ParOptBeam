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
            "density": 7850.0,
            "youngs_modulus": 2069000000,
            "poisson_ratio": 0.29,
            "damping_ratio": 0.05
        },
        "geometry": {
            "length_x": 1.2,
            "number_of_elements": 1,
            "defined_on_intervals": [{
                "interval_bounds": [0.0, "End"],
                "length_y": [1.0],
                "length_z": [1.0],
                "area": [0.0001],
                "shear_area_y": [0.0],
                "shear_area_z": [0.0],
                "moment_of_inertia_y": [0.0001],
                "moment_of_inertia_z": [0.0001],
                "torsional_moment_of_inertia": [0.0001],
                "outrigger_mass": [0.0],
                "outrigger_stiffness": [0.0]}]
        }
    },
    "boundary_conditions": "fixed-free"
}


def test_structure_model():
    beam = StraightBeam(params)
