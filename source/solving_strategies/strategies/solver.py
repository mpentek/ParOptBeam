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
import numpy as np


class Solver(object):
    def __init__(self, array_time, time_integration_scheme, dt, comp_model, initial_conditions, force):
        # vector of time
        self.array_time = array_time

        # time step
        self.dt = dt

        # scheme
        self.scheme = None

        # mass, damping and spring stiffness
        self.M = comp_model[0]
        self.B = comp_model[1]
        self.K = comp_model[2]

        # external forces
        self.force = force

        # placeholders for the solution
        rows = len(initial_conditions[0])
        cols = len(self.array_time)

        # adding additional attributes to the derived class
        self.displacement = np.zeros((rows, cols))
        self.velocity = np.zeros((rows, cols))
        self.acceleration = np.zeros((rows, cols))

        self._print_structural_setup()

    def _print_structural_setup(self):
        print("Printing structural setup in the solver base class:")
        print("mass: ", self.M)
        print("damping: ", self.B)
        print("stiffness: ", self.K)
        print(" ")

    def solve(self):
        pass