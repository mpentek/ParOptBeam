import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.model.structure_model import StraightBeam



params = {
    "name": "CaarcBeamPrototypeOptimizable",
    "domain_size": "3D",
    "system_parameters": {
        "element_type": "CRBeam",
        "material": {
            "density": 160.0,
            "youngs_modulus": 2.861e8,
            "poisson_ratio": 0.1,
            "damping_ratio": 0.005
        },
        "geometry": {
            "length_x": 180.0,
            "number_of_elements": 3,
            "defined_on_intervals": []
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
    schemes = ["ForwardEuler1", "BackwardEuler1"]
    beam = StraightBeam(params)

    for scheme in schemes:

        solver = ResidualBasedPicardSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, beam)
        solver.solve()
        plt.plot(array_time, solver.displacement[1, :], label=scheme)
        plt.legend()
    plt.show()

