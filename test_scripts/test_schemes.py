import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver

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


def test_linear_solver():
    for scheme in schemes:

        solver = LinearSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, None)
        solver.solve()
        plt.plot(array_time, solver.displacement[1, :], label=scheme)
        plt.legend()
    plt.show()


def test_picard_solver():
    schemes = ["ForwardEuler1", "BackwardEuler1"]

    for scheme in schemes:

        solver = ResidualBasedPicardSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, None)
        solver.solve()
        plt.plot(array_time, solver.displacement[1, :], label=scheme)
        plt.legend()
    plt.show()

