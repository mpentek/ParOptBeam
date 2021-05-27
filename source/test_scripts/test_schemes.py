import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import unittest

from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver


class test_schemes(unittest.TestCase):

    def test_schemes(self):
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
        for scheme in schemes:
            solver = LinearSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, None)
            solver.solve()
            plt.plot(array_time, solver.displacement[1, :], label=scheme)
            plt.legend()
        plt.show()


    def test_schemes_analytic_sdof(self):
        # M = np.array([[0.5, 0.0], [0.0, 1.0]])
        # B = np.array([[0.1, 0.0], [0.0, 0.1]])
        # K = np.array([[1.0, 0.0], [0.0, 2.0]])


        # initial conditions all zero --> homogeneous solution = 0
        u0 = np.array([0.0])
        v0 = np.array([0.0])
        a0 = np.array([0.0])

        # time setup
        dt = 0.01
        tend = 20.
        steps = int(tend / dt)
        array_time = np.linspace(0.0, tend, steps)

        # force setup
        freq_f = 1. # 1 [Hz]
        w_f = 2.*np.pi/freq_f
        p_0 = 10.
        f = np.array([p_0 * np.sin(w_f*array_time)])

        # system setup
        k = 1.
        c = 0.1
        m = 0.5
        w_n = np.sqrt(k/m)
        M = np.array([m])
        B = np.array([c])
        K = np.array([k])

        # analytic solution
        u_st = p_0/k
        ceta = c*w_n/(2.*k)
        rd = 1./(np.sqrt( np.square(1.-np.square(w_f/w_n)) + np.square(2.*ceta*(w_f/w_n))))
        phi = np.arctan((2*ceta*(w_f/w_n))/(1-np.square(w_f/w_n)))

        u_analytic = u_st*rd*np.cos(w_f*array_time-phi)

        # TODO: add homogeneous solution

        schemes = ["ForwardEuler1", "BackwardEuler1", "Euler12", "GenAlpha", "BDF2", "RungeKutta4"]
        for scheme in schemes:
            solver = LinearSolver(array_time, scheme, dt, [M,B,K], [u0, v0, a0], f, None)
            solver.solve()
            self.assertIsNone(np.testing.assert_allclose(u_analytic,solver.displacement[0],rtol = 1e-5, atol = 1e-10))
            
            if __name__ == "__main__":
                plt.figure()
                plt.plot(array_time,u_analytic,label='analytic')
                plt.plot(array_time,solver.displacement[0], label=scheme)
                plt.title(scheme)
                plt.show()


if __name__ == "__main__":
    unittest.main()