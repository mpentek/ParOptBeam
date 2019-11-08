# ===============================================================================
"""
        Derived classes from Solver

Created on:  15.10.2019
Last update: 16.10.2019
"""
# ===============================================================================
import numpy as np

from source.solving_strategies.strategies.solver import Solver
import source.auxiliary.global_definitions as GD


class ResidualBasedSolver(Solver):

    def __init__(self, array_time, time_integration_scheme, dt,
                 comp_model, initial_conditions, force, structure_model):
        super().__init__(array_time, time_integration_scheme, dt,
                         comp_model, initial_conditions, force, structure_model)

    def calculate_residual(self, q):
        pass

    def calculate_increment(self, ru):
        pass

    def solve_single_step(self):
        pass

    def get_displacement_from_element(self):
        u = np.zeros(self.structure_model.n_nodes * GD.DOFS_PER_NODE[self.structure_model.domain_size])

        for e in self.structure_model.elements:
            i_start = GD.DOFS_PER_NODE[e.domain_size] * e.index
            i_end = GD.DOFS_PER_NODE[e.domain_size] * e.index + GD.DOFS_PER_NODE[e.domain_size] * GD.NODES_PER_LEVEL
            u[i_start:i_end] += e.current_deformation

        u = self.structure_model.apply_bc_by_reduction(u, 'column_vector')
        return u

    def get_internal_force_from_element(self):
        q = np.zeros(self.structure_model.n_nodes * GD.DOFS_PER_NODE[self.structure_model.domain_size])

        for e in self.structure_model.elements:
            start_index = GD.DOFS_PER_NODE[e.domain_size] * e.index
            end_index = GD.DOFS_PER_NODE[e.domain_size] * e.index + GD.DOFS_PER_NODE[e.domain_size] * GD.NODES_PER_LEVEL

            q[start_index:end_index] += e.nodal_force_global

        q = self.structure_model.apply_bc_by_reduction(q, 'column_vector')
        return q

    def update_total(self, new_displacement):
        # updating displacement in the element
        for e in self.structure_model.elements:
            i_start = GD.DOFS_PER_NODE[e.domain_size] * e.index
            i_end = GD.DOFS_PER_NODE[e.domain_size] * e.index + GD.DOFS_PER_NODE[e.domain_size] * GD.NODES_PER_LEVEL
            new_displacement_e = new_displacement[i_start: i_end]
            e.update_total(new_displacement_e)

    def update_comp_model(self):
        self.structure_model.calculate_global_matrices()
        self.M = self.structure_model.comp_m
        self.B = self.structure_model.comp_b
        self.K = self.structure_model.comp_k
        self.scheme.update_comp_model([self.M, self.B, self.K])

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
            self.step = i
            current_time = self.array_time[i]
            print("time: {0:.2f}".format(current_time))

            self.solve_single_step()

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()
            self.dynamic_reaction[:, i] = self._compute_reaction()

            # update results
            self.scheme.update()
