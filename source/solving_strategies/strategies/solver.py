import numpy as np


class Solver(object):
    def __init__(self,
                 array_time, time_integration_scheme, dt,
                 comp_model,
                 initial_conditions,
                 force,
                 structure_model):
        # vector of time
        self.array_time = array_time

        # time step
        self.dt = dt

        # iteration
        self.step = 0

        # mass, damping and spring stiffness
        self.M = comp_model[0]
        self.B = comp_model[1]
        self.K = comp_model[2]

        # external forces
        self.force = force

        # for reaction calculation
        self.structure_model = structure_model

        # placeholders for the solution
        rows = len(initial_conditions[0])
        cols = len(self.array_time)

        # adding additional attributes to the derived class
        self.displacement = np.zeros((rows, cols))
        self.velocity = np.zeros((rows, cols))
        self.acceleration = np.zeros((rows, cols))
        self.dynamic_reaction = np.zeros((rows, cols))

        # initializing scheme
        self._init_scheme(time_integration_scheme,
                          comp_model, initial_conditions)

        self._print_structural_setup()
        self._print_solver_info()

    def _init_scheme(self, time_integration_scheme, comp_model, initial_conditions):
        if time_integration_scheme == "GenAlpha":
            from source.solving_strategies.schemes.generalized_alpha_scheme import GeneralizedAlphaScheme
            self.scheme = GeneralizedAlphaScheme(
                self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "Euler12":
            from source.solving_strategies.schemes.euler12_scheme import Euler12
            self.scheme = Euler12(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "ForwardEuler1":
            from source.solving_strategies.schemes.forward_euler1_scheme import ForwardEuler1
            self.scheme = ForwardEuler1(
                self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "BackwardEuler1":
            from source.solving_strategies.schemes.backward_euler1_scheme import BackwardEuler1
            self.scheme = BackwardEuler1(
                self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "RungeKutta4":
            from source.solving_strategies.schemes.runge_kutta4_scheme import RungeKutta4
            self.scheme = RungeKutta4(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "BDF2":
            from source.solving_strategies.schemes.bdf2_scheme import BDF2
            self.scheme = BDF2(self.dt, comp_model, initial_conditions)
        else:
            err_msg = "The requested time integration scheme \"" + time_integration_scheme
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"GenAlpha\", \"Euler12\", \"ForwardEuler1\", \"BackwardEuler1\", " \
                       "\"RungeKutta4\", \"BDF2\""
            raise Exception(err_msg)

    def _print_solver_info(self):
        pass

    def _print_structural_setup(self):
        print("Printing structural setup in the solver base class:")
        print("mass: ", self.M)
        print("damping: ", self.B)
        print("stiffness: ", self.K)
        print(" ")

    def solve(self):
        pass

    # def _compute_reaction(self):

    #     # TODO: check if this still correct in modal coordinates
    #     # if self.transform_into_modal:
    #     #     f1 = np.matmul(self.structure_model.recuperate_bc_by_extension(self.comp_m,axis='both'),
    #     #                    self.solver.acceleration)
    #     #     f2 = np.matmul(self.structure_model.recuperate_bc_by_extension(self.comp_b,axis='both'),
    #     #                    self.solver.velocity)
    #     #     f3 = np.matmul(self.structure_model.recuperate_bc_by_extension(self.comp_k,axis='both'),
    #     #                    self.solver.displacement)
    #     # else:
    #     u = self.displacement[:, self.step]
    #     v = self.velocity[:, self.step]
    #     a = self.acceleration[:, self.step]
    #     f1 = np.dot(self.M, a)
    #     f2 = np.dot(self.B, v)
    #     f3 = np.dot(self.K, u)
    #     dynamic_reaction = self.force[:, self.step] - f1 - f2 - f3

    #     # TODO: check if the treatment of elastic bc dofs is correct
    #     # TODO: check if this still applies in modal coordinates
    #     for dof_id, stiffness_val in self.structure_model.elastic_bc_dofs.items():
    #         # assuming a Rayleigh-model
    #         damping_val = stiffness_val * \
    #             self.structure_model.rayleigh_coefficients[1]

    #         f1 = 0.0 * a[dof_id]
    #         f2 = damping_val * v[dof_id]
    #         f3 = stiffness_val * u[dof_id]

    #         # overwrite the existing value with one solely from spring stiffness and damping
    #         dynamic_reaction[dof_id] = f1 + f2 + f3
    #     return dynamic_reaction
    
