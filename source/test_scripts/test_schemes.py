# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.test_utils.test_case import TestCase, TestMain


class TestSchemes(TestCase):

    def run_scheme(self, scheme: str) -> None:
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

        solver = LinearSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, None)
        solver.solve()

        reference_file_name = "test_schemes_" + scheme + ".csv"
        self.CompareToReferenceFile(
            solver.displacement[1,:],
            self.reference_directory / reference_file_name)


    def run_scheme_analytic_sdof(self, scheme: str) -> None:
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

        solver = LinearSolver(array_time, scheme, dt, [M,B,K], [u0, v0, a0], f, None)
        solver.solve()
        reference_file_name = "test_schemes_analytic_sdof_" + scheme + ".csv"
        self.CompareToReferenceFile(
            solver.displacement[0],
            self.reference_directory / reference_file_name,
            delta=1e-10)


    @TestCase.UniqueReferenceDirectory
    def test_ForwardEuler1(self) -> None:
        self.run_scheme("ForwardEuler1")
        self.run_scheme_analytic_sdof("ForwardEuler1")


    @TestCase.UniqueReferenceDirectory
    def test_BackwardEuler1(self) -> None:
        self.run_scheme("BackwardEuler1")
        self.run_scheme_analytic_sdof("BackwardEuler1")


    @TestCase.UniqueReferenceDirectory
    def test_Euler12(self) -> None:
        self.run_scheme("Euler12")
        self.run_scheme_analytic_sdof("Euler12")


    @TestCase.UniqueReferenceDirectory
    def test_GenAlpha(self) -> None:
        self.run_scheme("GenAlpha")
        self.run_scheme_analytic_sdof("GenAlpha")


    @TestCase.UniqueReferenceDirectory
    def test_BDF2(self) -> None:
        self.run_scheme("BDF2")
        self.run_scheme_analytic_sdof("BDF2")


    @TestCase.UniqueReferenceDirectory
    def test_RungeKutta4(self) -> None:
        self.run_scheme("RungeKutta4")
        self.run_scheme_analytic_sdof("RungeKutta4")


if __name__ == "__main__":
    TestMain()