import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from source.solving_strategies.strategies.residual_based_newton_raphson_solver import ResidualBasedNewtonRaphsonSolver
from source.model.structure_model import StraightBeam

default_cycler = (cycler(color=['b', 'b', 'b', 'g', 'r', 'k']) +
                  cycler(linestyle=['-', '--', ':', '-', '-', '-']))

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=default_cycler)

np.set_printoptions(suppress=False, precision=4)


params = {
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
beam = StraightBeam(params)

f_ext = beam.apply_bc_by_reduction(f_ext, 'column').T


def test_newton_raphson_solver():

    solver = ResidualBasedNewtonRaphsonSolver(array_time, scheme, dt,
                                       [beam.comp_m, beam.comp_b, beam.comp_k],
                                       [u0, v0, a0], f_ext, beam)

    solver.solve()

    reaction = solver.dynamic_reaction[:, 0]
    print("\nreaction = ", reaction)

    displacement = solver.scheme.get_displacement()
    displacement_kratos = np.array([1.0226219e-07, 0.0000000e+00, 2.7839541e-04,
                                    0.0000000e+00, -3.4799422e-04, 0.0000000e+00 ])
    print("\ndisplacement        = ", displacement)
    print("displacement_kratos = ", displacement_kratos)

    # new_displacement = solver.scheme.get_displacement()
    # new_displacement = solver.structure_model.recuperate_bc_by_extension(new_displacement, 'column_vector')
    # solver.update_total(new_displacement)
    # r = solver.calculate_residual(solver.force[:, 0])
    # print("ru = " + str(r))
    #
    # dp = solver.calculate_increment(r)
    # dp = solver.structure_model.recuperate_bc_by_extension(dp, 'column_vector')
    # print("\ndp =", dp)
    #
    # solver.update_incremental(dp)
    #
    # phi_s = solver.structure_model.elements[0]._calculate_symmetric_deformation_mode()
    # print("\nPhi_s = ", phi_s)
    # phi_s = solver.structure_model.elements[0].phi_s
    # print("Phi_s + d_phi_s = ", phi_s)
    # phi_s_kratos = np.array([0., 0., 0.])
    # print("Phi_s_kratos = ", phi_s_kratos)
    #
    # phi_a = solver.structure_model.elements[0]._calculate_antisymmetric_deformation_mode()
    # print("\nPhi_a = ", phi_a)
    # phi_a = solver.structure_model.elements[0].phi_a
    # print("Phi_a + d_phi_a = ", phi_a)
    # phi_a_kratos = np.array([0, 0.000463992, 0])
    # print("Phi_a_kratos = ", phi_a_kratos)
    #
    # v = solver.structure_model.elements[0].v
    # print("\nv = ", v)
    # v_kratos = np.array([0, 0, 0, 3.22933e-08, 0.000463992, 0])
    # print("v_kratos = ", v_kratos)

    # plt.plot(array_time, solver.displacement[1, :])
    # plt.show()
