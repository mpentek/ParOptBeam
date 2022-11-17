# --- External Imports ---
import numpy as np

# --- Internal Imports ---
from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.solving_strategies.strategies.residual_based_picard_solver import ResidualBasedPicardSolver
from source.model.structure_model import StraightBeam
from source.test_utils.test_case import TestCase, TestMain
from source.test_utils.code_structure import TEST_KRATOS_REFERENCE_RESULTS_DIRECTORY


class TestDynamicResidualBasedSolvers(TestCase):

    @TestCase.UniqueReferenceDirectory
    def test_residual_based_solvers(self):
        dt = 0.1
        tend = 10.
        steps = int(tend / dt)
        array_time = np.linspace(0.0, tend, steps + 1)
        array_time_kratos = np.linspace(0.1, 10, 101)

        f_ext = np.array([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 100.0 * np.sin(t), 0.0, 0.0, 0.0])
                        for t in np.sin(array_time)])

        u0 = np.zeros(6)
        v0 = np.zeros(6)
        a0 = np.zeros(6)

        scheme = "BackwardEuler1"
        beam = StraightBeam(self.params)

        f_ext = beam.apply_bc_by_reduction(f_ext, 'column').T

        newton_solver = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                                        [beam.comp_m, beam.comp_b, beam.comp_k],
                                                        [u0, v0, a0], f_ext, beam)

        picard_solver = ResidualBasedPicardSolver(array_time, scheme, dt,
                                                [beam.comp_m, beam.comp_b, beam.comp_k],
                                                [u0, v0, a0], f_ext, beam)

        newton_solver.solve()
        picard_solver.solve()

        self.CompareToReferenceFile(
            newton_solver.displacement[2,:],
            self.reference_directory / "dynamic_displacement_z_newton.csv")

        self.CompareToReferenceFile(
            picard_solver.displacement[2,:],
            self.reference_directory / "dynamic_displacement_z_picard.csv")

        if __name__ == "__main__":
            import matplotlib.pyplot as plt
            reference_file = TEST_KRATOS_REFERENCE_RESULTS_DIRECTORY / "dynamic_displacement_z.txt"
            disp_z_soln = np.loadtxt(reference_file)[:, 1]
            plt.plot(array_time, newton_solver.displacement[2, :], c='b', label='Newton Raphson')
            plt.plot(array_time, picard_solver.displacement[2, :], c='g', label='Picard')
            plt.plot(array_time_kratos, disp_z_soln, c='k', label='Kratos reference')
            plt.grid()
            plt.legend()
            plt.show()


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
                    "damping_ratio": 0.1
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