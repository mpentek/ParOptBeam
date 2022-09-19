# --- External Imports ---
import numpy

# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.test_utils.test_case import TestCase, TestMain


schemes_beam_parameters = {
    "name": "name",
    "domain_size": "3D",
    "system_parameters": {
        "element_params": {
            "type": "Bernoulli",
            "is_nonlinear": False
        },
        "material": {
            "is_nonlinear": False,
            "density": 7850.0,
            "youngs_modulus": 2.10e11,
            "poisson_ratio": 0.3,
            "damping_ratio": 0.0
        },
        "geometry": {
            "length_x": 25,
            "number_of_elements": 1,
            "defined_on_intervals": [{
                "interval_bounds" : [0.0,"End"],
                "length_y": [0.20],
                "length_z": [0.40],
                "area"    : [0.08],
                "shear_area_y" : [0.0667],
                "shear_area_z" : [0.0667],
                "moment_of_inertia_y" : [0.0010667],
                "moment_of_inertia_z" : [0.0002667],
                "torsional_moment_of_inertia" : [0.00007328]}]
        }
    },
    "boundary_conditions": "fixed-free"
}


class TestSchemes(TestCase):

    def test_AllSchemes(self) -> None:
        for number_of_elements in (1, 2):
            for scheme in ("ForwardEuler1", "BackwardEuler1", "Euler12", "GenAlpha", "BDF2", "RungeKutta4"):
                with self.subTest(msg = scheme):
                    for solver_type in (LinearSolver, ResidualBasedPicardSolver, ResidualBasedNewtonRaphsonSolver):
                        # Construct structural components
                        beam_parameters = schemes_beam_parameters.copy()
                        beam_parameters["system_parameters"]["geometry"]["number_of_elements"] = number_of_elements
                        model = StraightBeam(beam_parameters)

                        number_of_dofs = len(model.k)

                        # Time discretization
                        time_begin = 0.0
                        time_end = 1.0
                        time_step_size = int(1e-4)
                        discrete_time = numpy.linspace(time_begin, time_end, time_step_size)

                        number_of_steps = len(discrete_time)

                        # Initial conditions (free vibration from initial displacement)
                        initial_displacement = numpy.zeros(number_of_dofs)
                        initial_velocity = numpy.zeros(number_of_dofs)
                        initial_acceleration = numpy.zeros(number_of_dofs)
                        external_forces = numpy.zeros((number_of_dofs, number_of_steps))

                        initial_displacement[0] = 1.0

                        # Time integration
                        solver = solver_type(discrete_time,
                                            scheme,
                                            time_step_size,
                                            [model.m, model.b, model.k],
                                            [initial_displacement, initial_velocity, initial_acceleration],
                                            external_forces,
                                            model)

                        print(solver)

"""
class TestSchemes(TestCase):

    def run_scheme(self, scheme: str) -> None:
        M = numpy.array([[0.5, 0.0], [0.0, 1.0]])
        B = numpy.array([[0.1, 0.0], [0.0, 0.1]])
        K = numpy.array([[1.0, 0.0], [0.0, 2.0]])
        u0 = numpy.array([0.0, 1.0])
        v0 = numpy.array([0.0, 0.0])
        a0 = numpy.array([0.0, 0.0])
        dt = 0.01
        tend = 20.
        steps = int(tend / dt)
        array_time = numpy.linspace(0.0, tend, steps)
        f = numpy.array([0.0 * array_time, 0.6 * numpy.sin(array_time)])

        solver = LinearSolver(array_time, scheme, dt, [M, B, K], [u0, v0, a0], f, None)
        solver.solve()

        reference_file_name = "test_schemes_" + scheme + ".csv"
        self.CompareToReferenceFile(
            solver.displacement[1,:],
            self.reference_directory / reference_file_name)


    def run_scheme_analytic_sdof(self, scheme: str) -> None:
        # M = numpy.array([[0.5, 0.0], [0.0, 1.0]])
        # B = numpy.array([[0.1, 0.0], [0.0, 0.1]])
        # K = numpy.array([[1.0, 0.0], [0.0, 2.0]])


        # initial conditions all zero --> homogeneous solution = 0
        u0 = numpy.array([0.0])
        v0 = numpy.array([0.0])
        a0 = numpy.array([0.0])

        # time setup
        dt = 0.01
        tend = 20.
        steps = int(tend / dt)
        array_time = numpy.linspace(0.0, tend, steps)

        # force setup
        freq_f = 1. # 1 [Hz]
        w_f = 2.*numpy.pi/freq_f
        p_0 = 10.
        f = numpy.array([p_0 * numpy.sin(w_f*array_time)])

        # system setup
        k = 1.
        c = 0.1
        m = 0.5
        w_n = numpy.sqrt(k/m)
        M = numpy.array([m])
        B = numpy.array([c])
        K = numpy.array([k])

        # analytic solution
        #u_st = p_0/k
        #ceta = c*w_n/(2.*k)
        #rd = 1./(numpy.sqrt( numpy.square(1.-numpy.square(w_f/w_n)) + numpy.square(2.*ceta*(w_f/w_n))))
        #phi = numpy.arctan((2*ceta*(w_f/w_n))/(1-numpy.square(w_f/w_n)))

        #u_analytic = u_st*rd*numpy.cos(w_f*array_time-phi)

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
"""


if __name__ == "__main__":
    TestMain()