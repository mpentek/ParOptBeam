# --- External Imports ---
import numpy
import scipy.linalg

# --- Internal Imports ---
from source.model.structure_model import StraightBeam
from source.solving_strategies.strategies.linear_solver import LinearSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.test_utils.test_case import TestCase, TestMain
from source.auxiliary import global_definitions
from source.test_utils.analytical_solutions import EulerBernoulli as AnalyticalBeam

# --- STD Imports ---
import typing


schemes_beam_parameters = {
    "name": "Name",
    "domain_size": "3D",
    "boundary_conditions" : "fixed-free",
    "system_parameters": {
        "element_params": {
            "type": "Bernoulli",
            "is_nonlinear": False
        },
        "material": {
            "is_nonlinear": False,
            "density": 7850.0,
            "youngs_modulus": 210e9,
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
    }
}


class TestSchemes(TestCase):

    def GetAnalyticalSolution(self, initial_displacement: float) -> numpy.array:
        parameters = schemes_beam_parameters["system_parameters"]
        material = parameters["material"].copy()
        geometry = parameters["geometry"]
        geometry.update(parameters["geometry"]["defined_on_intervals"][0])

        section_density = material["density"] * geometry["area"][0]
        stiffness = material["youngs_modulus"]
        length = geometry["length_x"]
        moment_of_inertia = geometry["moment_of_inertia_" + ("y" if self.deflection_direction == "z" else "z")][0]

        analytical_beam = AnalyticalBeam(section_density, stiffness, length, moment_of_inertia)
        solution_functor = analytical_beam.GetDynamicSolution(initial_displacement, schemes_beam_parameters["boundary_conditions"].split("-"))
        return numpy.array([solution_functor(t, length) for t in self.time_samples]), solution_functor

    @property
    def initial_displacement(self) -> float:
        return 1e-3

    @property
    def time_step_size(self) -> float:
        return 5e-5

    @property
    def time_end(self) -> float:
        return 4e0

    @property
    def deflection_direction(self) -> str:
        return "y"

    @property
    def time_samples(self) -> numpy.array:
        time_begin = 0.0
        return numpy.linspace(time_begin, self.time_end, round((self.time_end-time_begin) / self.time_step_size) + 1)

    @property
    def tolerance(self) -> float:
        return 1e-1

    @property
    def time_integration_schemes(self) -> list["str"]:
        # return ("ForwardEuler1", "BackwardEuler1", "Euler12", "GenAlpha", "BDF2", "RungeKutta4")
        return ["BDF2"]

    @property
    def solvers(self) -> list["type"]:
        # return (LinearSolver, ResidualBasedPicardSolver, ResidualBasedNewtonRaphsonSolver)
        return [LinearSolver]

    @property
    def discretizations(self) -> typing.Generator:
        return range(2, 4)

    def test_AllSchemes(self) -> None:
        # Time discretization
        time_samples = self.time_samples
        number_of_steps = len(time_samples)

        analytical_solution, analytical_functor = self.GetAnalyticalSolution(self.initial_displacement)
        domain_size = schemes_beam_parameters["domain_size"]
        cases = {}

        # Loop through cases
        for number_of_elements in self.discretizations:
            # Construct model
            beam_parameters = schemes_beam_parameters.copy()
            beam_parameters["system_parameters"]["geometry"]["number_of_elements"] = number_of_elements
            model = StraightBeam(beam_parameters)

            # Compute static deformation for initial conditions
            # |                   |
            # |                   V
            # |--------------------
            # |
            load_dof_index = (number_of_elements - 1) * global_definitions.DOFS_PER_NODE[domain_size] + global_definitions.DOF_LABELS[domain_size].index(self.deflection_direction)
            load_vector = numpy.zeros(model.k.shape[0])
            load_vector[load_dof_index] = 1.0
            initial_displacement = scipy.linalg.solve(model.apply_bc_by_reduction(model.k), model.apply_bc_by_reduction(load_vector, axis="row_vector"))
            initial_displacement = numpy.ravel(model.recuperate_bc_by_extension(initial_displacement, axis="row_vector"))
            initial_displacement *= self.initial_displacement / initial_displacement[load_dof_index] # scale initial shape (assuming linear behaviour)

            # Initial conditions (free vibration from initial displacement)
            number_of_dofs = model.k.shape[0]
            initial_velocity = numpy.zeros(number_of_dofs)
            initial_acceleration = numpy.zeros(number_of_dofs)
            external_forces = numpy.zeros((number_of_dofs, number_of_steps))

            for scheme in self.time_integration_schemes:
                cases.setdefault(scheme, {})
                for solver_type in self.solvers:
                    solver_name = solver_type.__name__
                    cases[scheme].setdefault(solver_name, {})
                    with self.subTest(scheme=scheme, solver_name=solver_name, number_of_elements=number_of_elements):
                        try:
                            # Time integration
                            solver = solver_type(time_samples,
                                                 scheme,
                                                 self.time_step_size,
                                                 [model.comp_m, model.comp_b, model.comp_k],
                                                 [numpy.ravel(model.apply_bc_by_reduction(item, axis="row_vector")) for item in [initial_displacement, initial_velocity, initial_acceleration]],
                                                 model.apply_bc_by_reduction(external_forces, axis="row"),
                                                 model)

                            solver.solve()
                            displacement_history = model.recuperate_bc_by_extension(solver.displacement, axis="row")[load_dof_index,:]

                            # Error norm as normalized deviation
                            error = numpy.trapz(((analytical_solution - displacement_history) / numpy.max(numpy.abs(analytical_solution)))**2, time_samples) / (time_samples[-1] - time_samples[0])

                            # # --- Debug begin ---
                            # import numpy as np
                            # import matplotlib.pyplot as plt
                            # print(f"{scheme} {solver_name} {number_of_elements} {error}")
                            # plt.plot(time_samples, analytical_solution)
                            # plt.plot(time_samples, displacement_history)
                            # plt.show()
                            # plt.close()
                            # # --- Debug end ---

                            self.assertLess(error, self.tolerance)

                            cases[scheme][solver_name][number_of_elements] = "Pass"
                        except AssertionError as exception:
                            cases[scheme][solver_name][number_of_elements] = "Fail"
                            raise exception
                        except Exception as exception:
                            cases[scheme][solver_name][number_of_elements] = "Error"
                            raise exception

        # Dump results dict
        import json
        print(json.dumps(cases, indent=4))


if __name__ == "__main__":
    TestMain()
