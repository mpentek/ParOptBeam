# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain


# TODO (mate kelemen)
class TestStatcResiudalBasedSolvers(TestCase):

    @TestCase.UniqueReferenceDirectory
    def test_newton_raphson_solver(self):
        dt = 1.
        tend = 1.
        steps = int(tend / dt)
        array_time = np.linspace(0.0, tend, steps + 1)
        f_ext = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
        ])

        u0 = np.zeros(6)
        v0 = np.zeros(6)
        a0 = np.zeros(6)

        scheme = "BackwardEuler1"
        beam = StraightBeam(self.params)

        f_ext = beam.apply_bc_by_reduction(f_ext, 'column').T

        solver = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                                [beam.comp_m, beam.comp_b, beam.comp_k],
                                                [u0, v0, a0], f_ext, beam)

        solver.solve()

        reaction = solver.dynamic_reaction[:, 0]
        reaction_kratos = np.array([6.0974569e-09, -0.0000000e+00, -2.6529294e-05,
                                    -0.0000000e+00, -2.1223438e-05, -0.0000000e+00])

        #self.assertArrayAlmostEqual(reaction, reaction_kratos)

        displacement = solver.scheme.get_displacement()
        displacement_kratos = np.array([1.0226219e-07, 0.0000000e+00, 2.7839541e-04,
                                        0.0000000e+00, -3.4799422e-04, 0.0000000e+00])

        #self.assertArrayAlmostEqual(displacement, displacement_kratos)

        if __name__ == "__main__":
            from matplotlib import pyplot
            pyplot.plot(reaction, label="BackwardEuler1")
            pyplot.plot(reaction_kratos, label="Kratos")
            pyplot.grid()
            pyplot.legend()
            pyplot.show()


    @property
    def params(self):
        return {
            "name": "CaarcBeamPrototypeOptimizable",
            "domain_size": "3D",
            "system_parameters": {
                "element_params": {
                    "type": "CRBeam",
                    "is_nonlinear": True
                },
                "material": {
                    "density": 7850.0,
                    "youngs_modulus": 2069000000,
                    "poisson_ratio": 0.29,
                    "damping_ratio": 0.05
                },
                "geometry": {
                    "length_x": 1.2,
                    "number_of_elements": 1,
                    "defined_on_intervals": [{
                        "interval_bounds": [0.0, "End"],
                        "length_y": [1.0],
                        "length_z": [1.0],
                        "area": [0.0001],
                        "shear_area_y": [0.0],
                        "shear_area_z": [0.0],
                        "moment_of_inertia_y": [0.0001],
                        "moment_of_inertia_z": [0.0001],
                        "torsional_moment_of_inertia": [0.0001],
                        "outrigger_mass": [0.0],
                        "outrigger_stiffness": [0.0]}]
                }
            },
            "boundary_conditions": "fixed-free"
        }


if __name__ == "__main__":
    TestMain()