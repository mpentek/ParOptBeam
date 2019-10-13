import numpy as np
from source.scheme.euler12_scheme import Euler12
from source.scheme.bdf2_scheme import BDF2
from source.scheme.runge_kutta4_scheme import RungeKutta4


def test():
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[0.0, 0.0], [0.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    u0 = np.array([0.0, 1.0])
    v0 = np.array([0.0, 0.0])
    a0 = np.array([0.0, 0.0])
    dt = 0.01
    steps = 1000
    f1 = np.array([0.0, 0.0])

    euler12_displacement = np.empty([2, steps])
    euler12_velocity = np.empty([2, steps])
    euler12_acceleration = np.empty([2, steps])
    euler12_solver = Euler12(dt, [M, B, K], [u0, v0, a0])

    bdf2_displacement = np.empty([2, steps])
    bdf2_velocity = np.empty([2, steps])
    bdf2_acceleration = np.empty([2, steps])
    bdf2_solver = BDF2(dt, [M, B, K], [u0, v0, a0])

    rk4_displacement = np.empty([2, steps])
    rk4_velocity = np.empty([2, steps])
    rk4_acceleration = np.empty([2, steps])
    rk4_solver = RungeKutta4(dt, [M, B, K], [u0, v0, a0])

    for i in range(0, steps):

        euler12_solver.solve_structure(f1)
        bdf2_solver.solve_structure(f1)
        rk4_solver.solve_structure(f1)

        # appending results to the list
        euler12_displacement[:, i] = euler12_solver.get_displacement()
        euler12_velocity[:, i] = euler12_solver.get_velocity()
        euler12_acceleration[:, i] = euler12_solver.get_acceleration()

        bdf2_displacement[:, i] = bdf2_solver.get_displacement()
        bdf2_velocity[:, i] = bdf2_solver.get_velocity()
        bdf2_acceleration[:, i] = bdf2_solver.get_acceleration()

        rk4_displacement[:, i] = rk4_solver.get_displacement()
        rk4_velocity[:, i] = rk4_solver.get_velocity()
        rk4_acceleration[:, i] = rk4_solver.get_acceleration()

        # update results
        euler12_solver.update_structure_time_step()
        bdf2_solver.update_structure_time_step()
        rk4_solver.update_structure_time_step()

    import matplotlib.pyplot as plt
    plt.plot(euler12_displacement[1, :], label='euler12', color='b')
    plt.plot(bdf2_displacement[1, :], label='bdf2', color='r')
    plt.plot(rk4_displacement[1, :], label='rk4', color='g')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()