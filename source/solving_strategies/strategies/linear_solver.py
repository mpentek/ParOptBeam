# ===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18 
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
        Time integration scheme base class and derived classes for specific implementations

Author: mate.pentek@tum.de, anoop.kodakkal@tum.de, klaus.sautter@tum.de


      
Note:   UPDATE: The script has been written using publicly available information and 
        data, use accordingly. It has been written and tested with Python 2.7.9.
        Tested and works also with Python 3.4.3 (already see differences in print).
        Module dependencies (-> line 61-74): 
            python
            numpy
            sympy
            matplotlib.pyplot

Created on:  15.10.2017
Last update: 15.10.2019
'''
# ===============================================================================

from source.solving_strategies.strategies.solver import Solver
from source.solving_strategies.schemes.generalized_alpha_scheme import GeneralizedAlphaScheme
from source.solving_strategies.schemes.euler12_scheme import Euler12
from source.solving_strategies.schemes.forward_euler1_scheme import ForwardEuler1
from source.solving_strategies.schemes.backward_euler1_scheme import BackwardEuler1
from source.solving_strategies.schemes.runge_kutta4_scheme import RungeKutta4
from source.solving_strategies.schemes.bdf2_scheme import BDF2


class LinearSolver(Solver):
    def __init__(self, array_time, time_integration_scheme, dt, comp_model, initial_conditions, force):

        super().__init__(array_time, time_integration_scheme, dt, comp_model, initial_conditions, force)

        if time_integration_scheme == "GenAlpha":
            self.scheme = GeneralizedAlphaScheme(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "Euler12":
            self.scheme = Euler12(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "ForwardEuler1":
            self.scheme = ForwardEuler1(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "BackwardEuler1":
            self.scheme = BackwardEuler1(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "RungeKutta4":
            self.scheme = RungeKutta4(self.dt, comp_model, initial_conditions)
        elif time_integration_scheme == "BDF2":
            self.scheme = BDF2(self.dt, comp_model, initial_conditions)
        else:
            err_msg = "The requested time integration scheme \"" + time_integration_scheme
            err_msg += "\" is not available \n"
            err_msg += "Choose one of: \"GenAlpha\", \"Euler12\", \"ForwardEuler1\", \"BackwardEuler1\", " \
                       "\"RungeKutta4\", \"BDF2\""
            raise Exception(err_msg)

    def solve(self):
        # time loop
        for i in range(0, len(self.array_time)):
            current_time = self.array_time[i]
            print("time: ", "{0:.2f}".format(current_time))
            self.scheme.solve_single_step(self.force[:, i])

            # appending results to the list
            self.displacement[:, i] = self.scheme.get_displacement()
            self.velocity[:, i] = self.scheme.get_velocity()
            self.acceleration[:, i] = self.scheme.get_acceleration()

            # update results
            self.scheme.update_structure_time_step()
