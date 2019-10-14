import numpy as np
from source.scheme.euler12_scheme import Euler12
from source.scheme.bdf2_scheme import BDF2
from source.scheme.runge_kutta4_scheme import RungeKutta4
from source.scheme.generalized_alpha_scheme import GeneralizedAlphaScheme


def test():
    M = np.array([[0.5, 0.0], [0.0, 1.0]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    K = np.array([[1.0, 0.0], [0.0, 2.0]])
    u0 = np.array([0.0, 1.0])
    v0 = np.array([0.0, 0.0])
    a0 = np.array([0.0, 0.0])
    dt = 0.05
    tend = 20.
    steps = int(tend / dt)
    f = np.array([0.0, 0.6])

    euler12_displacement = np.empty([2, steps])
    euler12_solver = Euler12(dt, [M, B, K], [u0, v0, a0])

    genalpha_displacement = np.empty([2, steps])
    genalpha_solver = GeneralizedAlphaScheme(dt, [M, B, K], [u0, v0, a0])

    bdf2_displacement = np.empty([2, steps])
    bdf2_solver = BDF2(dt, [M, B, K], [u0, v0, a0])

    rk4_displacement = np.empty([2, steps])
    rk4_solver = RungeKutta4(dt, [M, B, K], [u0, v0, a0])

    for i in range(0, steps):
        t = i*dt
        f1 = np.sin(t) * f
        euler12_solver.solve_structure(f1)
        genalpha_solver.solve_structure(f1)
        bdf2_solver.solve_structure(f1)
        rk4_solver.solve_structure(f1)

        # appending results to the list
        euler12_displacement[:, i] = euler12_solver.get_displacement()
        genalpha_displacement[:, i] = genalpha_solver.get_displacement()
        bdf2_displacement[:, i] = bdf2_solver.get_displacement()
        rk4_displacement[:, i] = rk4_solver.get_displacement()

        # update results
        euler12_solver.update_structure_time_step()
        genalpha_solver.update_structure_time_step()
        bdf2_solver.update_structure_time_step()
        rk4_solver.update_structure_time_step()

    t = np.asarray(range(0, steps)) * dt
    import matplotlib.pyplot as plt
    plt.plot(t, euler12_displacement[1, :], label='Euler12', color='k')
    plt.plot(t, genalpha_displacement[1, :], label='GenAlpha', color='b')
    plt.plot(t, bdf2_displacement[1, :], label='bdf2', color='r')
    plt.plot(t, rk4_displacement[1, :], label='rk4', color='g')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
