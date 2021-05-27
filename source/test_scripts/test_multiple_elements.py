import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.model.structure_model import StraightBeam

if __name__ == "__main__":
    np.set_printoptions(suppress=False, precision=2, linewidth=140)

    params_one_element = {
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
                "damping_ratio": 0.
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

    params_two_element = {
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
                "damping_ratio": 0.
            },
            "geometry": {
                "length_x": 1.2,
                "number_of_elements": 2,
                "defined_on_intervals": [{
                    "interval_bounds": [0.0, 0.6],
                    "length_y": [1.0],
                    "length_z": [1.0],
                    "area": [0.0001],
                    "shear_area_y": [0.0],
                    "shear_area_z": [0.0],
                    "moment_of_inertia_y": [0.0001],
                    "moment_of_inertia_z": [0.0001],
                    "torsional_moment_of_inertia": [0.0001],
                    "outrigger_mass": [0.0],
                    "outrigger_stiffness": [0.0]},
                    {
                        "interval_bounds": [0.6, "End"],
                        "length_y": [1.0],
                        "length_z": [1.0],
                        "area": [0.0001],
                        "shear_area_y": [0.0],
                        "shear_area_z": [0.0],
                        "moment_of_inertia_y": [0.0001],
                        "moment_of_inertia_z": [0.0001],
                        "torsional_moment_of_inertia": [0.0001],
                        "outrigger_mass": [0.0],
                        "outrigger_stiffness": [0.0]}
                ]
            }
        },
        "boundary_conditions": "fixed-free"
    }

    dt = 0.1
    tend = 10.
    steps = int(tend / dt)
    array_time = np.linspace(0.0, tend, steps + 1)
    array_time_kratos = np.linspace(0.1, 10, 101)


    def test_multiple_elements():
        f_ext_one_element = np.array([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 100.0 * np.sin(t), 0.0, 0.0, 0.0])
                                    for t in np.sin(array_time)])

        f_ext_two_element = np.array([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 100.0 * np.sin(t), 0.0, 0.0, 0.0])
                                    for t in np.sin(array_time)])

        u0_1 = np.zeros(6)
        v0_1 = np.zeros(6)
        a0_1 = np.zeros(6)

        u0_2 = np.zeros(12)
        v0_2 = np.zeros(12)
        a0_2 = np.zeros(12)

        scheme = "BackwardEuler1"
        beam_1 = StraightBeam(params_one_element)
        beam_2 = StraightBeam(params_two_element)

        f_ext_1 = beam_1.apply_bc_by_reduction(f_ext_one_element, 'column').T
        f_ext_2 = beam_2.apply_bc_by_reduction(f_ext_two_element, 'column').T

        solver_1 = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                                    [beam_1.comp_m, beam_1.comp_b, beam_1.comp_k],
                                                    [u0_1, v0_1, a0_1], f_ext_1, beam_1)

        solver_2 = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                                    [beam_2.comp_m, beam_2.comp_b, beam_2.comp_k],
                                                    [u0_2, v0_2, a0_2], f_ext_2, beam_2)

        solver_1.solve()
        solver_2.solve()

        reference_file = "kratos_reference_results/dynamic_displacement_z.txt"
        disp_z_soln = np.loadtxt(reference_file)[:, 1]

        plt.plot(array_time, solver_1.displacement[2, :], c='b', label='one element')
        plt.plot(array_time, solver_2.displacement[2, :], c='g', label='two elements')
        plt.plot(array_time_kratos, disp_z_soln, c='k', label='kratos reference')
        plt.grid()
        plt.legend()
        plt.show()
