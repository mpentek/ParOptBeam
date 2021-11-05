# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.test_utils.test_case import TestCase, TestMain
from source.test_utils.code_structure import TEST_REFERENCE_OUTPUT_DIRECTORY


class test_schemes(TestCase):

    def test_schemes(self):
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

            reference_file_name = "test_schemes_" + scheme + ".csv"
            self.CompareToReferenceFile(
                solver.displacement[1,:],
                self.reference_directory / reference_file_name)


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
        #u_st = p_0/k
        #ceta = c*w_n/(2.*k)
        #rd = 1./(np.sqrt( np.square(1.-np.square(w_f/w_n)) + np.square(2.*ceta*(w_f/w_n))))
        #phi = np.arctan((2*ceta*(w_f/w_n))/(1-np.square(w_f/w_n)))

        #u_analytic = u_st*rd*np.cos(w_f*array_time-phi)

        # TODO: add homogeneous solution

        schemes = ["ForwardEuler1", "BackwardEuler1", "Euler12", "GenAlpha", "BDF2", "RungeKutta4"]
        for scheme in schemes:
            solver = LinearSolver(array_time, scheme, dt, [M,B,K], [u0, v0, a0], f, None)
            solver.solve()
            reference_file_name = "test_schemes_analytic_sdof_" + scheme + ".csv"
            self.CompareToReferenceFile(
                solver.displacement[0],
                self.reference_directory / reference_file_name,
                delta=1e-10)


    @property
    def reference_directory(self):
        return TEST_REFERENCE_OUTPUT_DIRECTORY / "test_schemes"


if __name__ == "__main__":
    TestMain()