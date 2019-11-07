import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.model.structure_model import StraightBeam


np.set_printoptions(suppress=False, precision=2, linewidth=140)


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

dt = 0.1
tend = 10.
steps = int(tend / dt)
array_time = np.linspace(0.0, tend, steps + 1)
array_time_kratos = np.linspace(0.1, 10, 101)


def test_newton_raphson_solver():
    f_ext = np.array([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 100.0 * np.sin(t), 0.0, 0.0, 0.0])
                     for t in np.sin(array_time)])

    u0 = np.zeros(6)
    v0 = np.zeros(6)
    a0 = np.zeros(6)

    scheme = "BackwardEuler1"
    beam = StraightBeam(params)

    f_ext = beam.apply_bc_by_reduction(f_ext, 'column').T

    solver = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                              [beam.comp_m, beam.comp_b, beam.comp_k],
                                              [u0, v0, a0], f_ext, beam)

    solver.solve()

    reference_file = "kratos_reference_results/dynamic_displacement_z.txt"
    disp_z_soln = np.loadtxt(reference_file)[:, 1]

    plt.plot(array_time, solver.displacement[2, :], c='b', label='solver')
    plt.plot(array_time_kratos, disp_z_soln, c='k', label='kratos reference')
    plt.grid()
    plt.legend()
    plt.show()

