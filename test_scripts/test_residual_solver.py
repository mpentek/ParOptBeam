import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.model.structure_model import StraightBeam


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
            "length_x": 100.0,
            "number_of_elements": 1,
            "defined_on_intervals": [{
                "interval_bounds": [0.0, 100.0],
                "length_y": [50.0],
                "length_z": [35.0],
                "area": [1750.0],
                "shear_area_y": [1460.0],
                "shear_area_z": [1460.0],
                "moment_of_inertia_y": [178646.0],
                "moment_of_inertia_z": [364583.0],
                "torsional_moment_of_inertia": [420175.0],
                "outrigger_mass": [0.0],
                "outrigger_stiffness": [0.0]}]
        }
    },
    "boundary_conditions": "fixed-free"
}

default_cycler = (cycler(color=['b', 'b', 'b', 'g', 'r', 'k']) +
                  cycler(linestyle=['-', '--', ':', '-', '-', '-']))

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=default_cycler)

M = np.array([[0.5, 0.0], [0.0, 1.0]])
B = np.array([[0.1, 0.0], [0.0, 0.1]])
K = np.array([[1.0, 0.0], [0.0, 2.0]])
u0 = np.array([0.0, 1.0])
v0 = np.array([0.0, 0.0])
a0 = np.array([0.0, 0.0])
dt = 0.01
tend = 20.
steps = int(tend / dt)
array_time = np.linspace(0.0, tend, steps)
f = np.array([0.0 * array_time, 0.6 * np.sin(array_time)])
schemes = ["ForwardEuler1", "BackwardEuler1", "Euler12", "GenAlpha", "BDF2", "RungeKutta4"]


def test_picard_solver():
    scheme = "BackwardEuler1"
    beam = StraightBeam(params)

    solver = ResidualBasedPicardSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, beam)
    solver.solve()
    plt.plot(array_time, solver.displacement[1, :])
    plt.show()
