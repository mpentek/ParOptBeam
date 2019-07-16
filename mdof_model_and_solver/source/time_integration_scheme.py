#===============================================================================
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

Created on:  22.11.2017
Last update: 09.07.2019
'''
#===============================================================================

import numpy as np


class TimeIntegrationScheme(object):
    def __init__(self, dt, structure, initial_conditions):
        # time step
        self.dt = dt
        
        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)

        # initial displacement, velocity and acceleration
        self.u1 = self.u0
        self.v1 = self.v0
        self.a1 = self.a0

        # force from a previous time step (initial force)
        self.f0 = None
        self.f1 = None
        
        self._print_structural_setup()
        self._print_time_integration_setup()

    def _print_structural_setup(self):
        print("Printing structural setup in time integration scheme base class:")
        print("mass: ", self.M)
        print("damping: ", self.B)
        print("stiffness: ", self.K)
        print(" ")

    def _print_time_integration_setup(self):
        pass

   
    def predict_displacement(self):
        return 2.0 * self.u1 - self.u0

    def solve_structure(self, ext_force):
        pass
 
    def print_values_at_current_step(self, n):
        print("Printing values at step no: ", n, " (+1)")
        print("u0: ", self.u1)
        print("v0: ", self.v1)
        print("a0: ", self.a1)
        print("f0: ", self.f1)
        print(" ")
        		
    def get_displacement(self):
        return self.u1

    def get_velocity(self):
        return self.v1
    
    def get_acceleration(self):
        return self.a1

    def get_old_displacement(self):
        return self.u0

    def get_old_velocity(self):
        return self.v0
    
    def get_old_acceleration(self):
        return self.a0
    
class GeneralizedAlphaScheme(TimeIntegrationScheme):

    def __init__(self, dt, structure, initial_conditions,  p_inf=0.16):     
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt
        
        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)
			             
        # generalized alpha parameters (to ensure unconditional stability, 2nd order accuracy)
        self.alphaM = (2.0 * p_inf - 1.0) / (p_inf + 1.0)
        self.alphaF = p_inf / (p_inf + 1.0)
        self.beta = 0.25 * (1 - self.alphaM + self.alphaF)**2
        self.gamma = 0.5 - self.alphaM + self.alphaF

        # coefficients for LHS
        self.a1h = (1.0 - self.alphaM) / (self.beta * self.dt**2)
        self.a2h = (1.0 - self.alphaF) * self.gamma / (self.beta * self.dt)
        self.a3h = 1.0 - self.alphaF

        # coefficients for mass
        self.a1m = self.a1h
        self.a2m = self.a1h * self.dt
        self.a3m = (1.0 - self.alphaM - 2.0 * self.beta) / (2.0 * self.beta)
        
        #coefficients for damping
        self.a1b = (1.0 - self.alphaF) * self.gamma / (self.beta * self.dt)
        self.a2b = (1.0 - self.alphaF) * self.gamma / self.beta - 1.0
        self.a3b = (1.0 - self.alphaF) * (0.5 * self.gamma / self.beta - 1.0) * self.dt
        
        # coefficient for stiffness
        self.a1k = -1.0 * self.alphaF
        
        # coefficients for velocity update
        self.a1v = self.gamma / (self.beta * self.dt)
        self.a2v = 1.0 - self.gamma / self.beta
        self.a3v = (1.0 - self.gamma / (2 * self.beta)) * self.dt
        
        # coefficients for acceleration update
        self.a1a = self.a1v / (self.dt * self.gamma)
        self.a2a = -1.0 / (self.beta * self.dt)
        self.a3a = 1.0 - 1.0 / (2.0 * self.beta)   
        
		#structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]
        # initial displacement, velocity and acceleration
        self.u1 = self.u0
        self.v1 = self.v0
        self.a1 = self.a0
			  
		# force from a previous time step (initial force)
        self.f0 = np.dot(self.M,self.a0) + np.dot(self.B,self.v0) + np.dot(self.K,self.u0)
        self.f1 = np.dot(self.M,self.a1) + np.dot(self.B,self.v1) + np.dot(self.K,self.u1)
        
        self._print_structural_setup()
        self._print_time_integration_setup()
    
    def _print_time_integration_setup(self):
        print("Printing Generalized Alpha Method integration scheme setup:")
        print("dt: ", self.dt)
        print("alphaM: ", self.alphaF)
        print("alphaF: ", self.alphaM)
        print("gamma: ", self.gamma)
        print("beta: ", self.beta)
        print(" ")

    def solve_structure(self, f1):
        
        F = (1.0 - self.alphaF) * f1 + self.alphaF * self.f0
		
        LHS = self.a1h * self.M + self.a2h * self.B + self.a3h * self.K 
        RHS = np.dot(self.M,(self.a1m * self.u0 + self.a2m * self.v0 + self.a3m * self.a0))
        RHS += np.dot(self.B,(self.a1b * self.u0 + self.a2b * self.v0 + self.a3b * self.a0)) 
        RHS += np.dot(self.a1k * self.K, self.u0) + F

        # update self.f1
        self.f1 = f1
        
        # updates self.u1,v1,a1
        self.u1 = np.linalg.solve(LHS, RHS)
        self.v1 = self.a1v * (self.u1 - self.u0) + self.a2v * self.v0 + self.a3v * self.a0
        self.a1 = self.a1a * (self.u1 - self.u0) + self.a2a * self.v0 + self.a3a * self.a0

    def update_structure_time_step(self):
        # update displacement, velocity and acceleration 
        self.u0 = self.u1
        self.v0 = self.v1
        self.a0 = self.a1
        
        # update the force   
        self.f0 = self.f1

class ForwardEuler1(TimeIntegrationScheme):
    """
    (Explicit) Forward Euler 1st order approximation 
    
    """
    def __init__(self, dt, structure, initial_conditions):     
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt
        
        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)
        
		#structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]
		        
        # initial values for time integration
        self.un1 = self.u0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()
    
    def _print_time_integration_setup(self):
        print("Printing (Explicit) Forward Euler 1 st order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):

        
        # calculates self.un0,vn0,an0
        self.un0 = self.un1 + self.dt * self.vn1
        self.vn0 = self.vn1+ self.dt * self.an1
        
        LHS = self.M
        RHS = f1 - np.dot(self.B, self.vn0) - np.dot(self.K, self.un0)  
        self.an0 =  np.linalg.solve(LHS, RHS)

        #update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un1 = self.un0
        self.vn1 = self.vn0
        self.an1 = self.an0

class BackwardEuler1(TimeIntegrationScheme):
    """
    (Implicit) Backward Euler 1st order approximation
    
    """
    def __init__(self, dt, structure, initial_conditions):     
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt
        
        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)
        
		#structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]
           
        # initial values for time integration
        self.un1 = self.u0
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()
    
    def _print_time_integration_setup(self):
        print("Printing (Implicit) Backward Euler 1 st order approximation integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):

        # calculates self.un0,vn0,an0
        LHS = self.M + np.dot(self.B, self.dt) + np.dot(self.K, self.dt ** 2)
        RHS = f1 - np.dot(self.B, self.vn1) - np.dot(self.K, self.un1)  - np.dot(self.K, self.vn1 * self.dt)
        self.an0 =  np.linalg.solve(LHS, RHS)
        self.vn0 = self.vn1 + self.dt * self.an0
        self.un0 = self.un1+ self.dt * self.vn0
        
        #update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un1 = self.un0
        self.vn1 = self.vn0
        self.an1 = self.an0
    
class Euler12(TimeIntegrationScheme):
    """
    Euler 1st and 2nd order 

    """
    def __init__(self, dt, structure, initial_conditions):     
        # introducing and initializing properties and coefficients
        # construct an object self with the input arguments dt, M, B, K,
        # pInf, u0, v0, a0

        # time step
        self.dt = dt
        
        # mass, damping and spring stiffness
        self.M = structure.apply_bc_by_reduction(structure.m)
        self.B = structure.apply_bc_by_reduction(structure.b)
        self.K = structure.apply_bc_by_reduction(structure.k)
        
		#structure
        # initial displacement, velocity and acceleration
        self.u0 = initial_conditions[0]
        self.v0 = initial_conditions[1]
        self.a0 = initial_conditions[2]
		        
        # initial values for time integration
        self.un2 = self.u0
        self.un1 = self.u0 - self.v0 * self.dt + self.a0 * (self.dt ** 2 / 2)
        self.vn1 = self.v0
        self.an1 = self.a0

        self._print_structural_setup()
        self._print_time_integration_setup()
    
    def _print_time_integration_setup(self):
        print("Printing Euler 1st and 2nd order method integration scheme setup:")
        print("dt: ", self.dt)
        print(" ")

    def solve_structure(self, f1):
        LHS = self.M + np.dot(self.B, self.dt / 2)
        RHS = f1 * self.dt ** 2 + np.dot(self.un1,(2 * self.M - self.K * self.dt **2)) 
        RHS += np.dot (self.un2, (-self.M + self.B * self.dt/2))
        
        # calculates self.un0,vn0,an0
        self.un0 = np.linalg.solve(LHS, RHS)
        self.vn0 = (self.un0 - self. un2) / 2 / self.dt  
        self.an0 = (self.un0 - 2 * self.un1 + self.un2) / self.dt ** 2 

        #update self.u1,v1,a1
        self.u1 = self.un0
        self.v1 = self.vn0
        self.a1 = self.an0

    def update_structure_time_step(self):
        # update self.un2 un1
        self.un2 = self.un1
        self.un1 = self.un0
        