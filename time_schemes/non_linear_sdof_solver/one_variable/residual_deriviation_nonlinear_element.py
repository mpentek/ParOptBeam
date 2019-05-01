################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f rewrite 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = ( f - C * v(t) - K * u(t) ) / M = f(t, u, v) = rhs
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################


import matplotlib.pyplot as plt
import numpy as np
from math import *
from sympy import *
import cmath
import sys

init_printing(use_unicode=True)

class SDoF:

    def __init__(self, time_scheme=None, numerical_scheme=None, K=1.0, M=1, C=0.1, f=0.1, u0=1.0, v0=0.0, dt=0.1):

        self.K = lambda u: (u**2 + 1)
        self.Ku = lambda u: self.K(u) * u
        self.C = C
        self.f = f
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.time_scheme = time_scheme
        self.numerical_scheme = numerical_scheme
        self.tend = 20.0
        self.print_scheme()


    def rhs(self, t, u, v):
        f, C, M = symbols('f C M')
        return (f - self.K(u) * u - C * v) / M

    # Function for the u'(t) = v ODE.
    def g(self, v):
        return v


    def euler_disp_based(self):
        # ### euler ###
        # v_n+1 = v_n + dt f(tn, v_n)
        print("##### euler #####")

        a_n1, a_n, u_n2, u_n1, u_n, u_nm1, u_nm2, u_nm3, v_n1, v_n, v_nm1, v_nm2, t, dt = symbols('a_n1 a_n u_n2 u_n1 u_n u_nm1 u_nm2 u_nm3 v_n1 v_n v_nm1 v_nm2 t dt')
        f, C, M = symbols('f C M')
        du, ru = symbols('du ru')

        v_n = (u_n1 - u_n) / dt
        v_nm1 = ( u_n - u_nm1) / dt
        a_nm1 = ( v_n - v_nm1 ) / dt

        r_u = f - (M * a_nm1 + C * v_nm1 + self.K(u_n) * u_nm1 )
        print("ru = ", r_u)
        
        drudu = diff(r_u, u_n1)
        eq_u = ru + drudu * du
        sol = solve(eq_u, du)
        du = (sol[0])

        print("du = ", du )


    def bdf1_disp_based(self):
        # ### BDF1 ###
        # v_n+1 = v_n + dt f(tn+1, v_n+1)
        print("##### BDF1 #####")

        a_n1, a_n, u_n2, u_n1, u_n, u_nm1, u_nm2, u_nm3, v_n1, v_n, v_nm1, v_nm2, t, dt = symbols('a_n1 a_n u_n2 u_n1 u_n u_nm1 u_nm2 u_nm3 v_n1 v_n v_nm1 v_nm2 t dt')
        f, C, M = symbols('f C M')
        du, ru = symbols('du ru')

        v_n1 = ( u_n1 - u_n) / dt
        v_n = ( u_n - u_nm1) / dt
        a_n1 = ( v_n1 - v_n ) / dt

        if self.numerical_scheme == 'Newton Raphson':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n1) * u_n1 )

        elif self.numerical_scheme == 'Picard':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n) * u_n1 )

        print("ru = ", r_u)

        drudu = diff(r_u, u_n1)
        eq_u = ru + drudu * du
        sol = solve(eq_u, du)
        du = (sol[0])

        print("du = ", du )


    def bdf1_vel_based(self):
        # ### BDF1 ###
        # v_n+1 = v_n + dt f(tn+1, v_n+1)
        print("##### BDF1 #####")

        a_n1, a_n, u_n2, u_n1, u_n, u_nm1, u_nm2, u_nm3, v_n1, v_n, v_nm1, v_nm2, t, dt = symbols('a_n1 a_n u_n2 u_n1 u_n u_nm1 u_nm2 u_nm3 v_n1 v_n v_nm1 v_nm2 t dt')
        f, C, M = symbols('f C M')
        du, ru = symbols('du ru')

        u_n1 = u_n + v_n1 * dt
        a_n1 = (v_n1 - v_n) / dt

        if self.numerical_scheme == 'Newton Raphson':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n1) * u_n1 )

        elif self.numerical_scheme == 'Picard':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n) * u_n1 )

        print("ru = ", r_u)

        drudu = diff(r_u, u_n1)
        eq_u = ru + drudu * du
        sol = solve(eq_u, du)
        du = (sol[0])

        print("du = ", du )


    def bdf2_disp_based(self):
        # ### BDF2 ###
        # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
        print("##### BDF2 #####")
        a_n1, a_n, u_n2, u_n1, u_n, u_nm1, u_nm2, u_nm3, v_n1, v_n, v_nm1, v_nm2, t, dt = symbols('a_n1 a_n u_n2 u_n1 u_n u_nm1 u_nm2 u_nm3 v_n1 v_n v_nm1 v_nm2 t dt')
        f, C, M = symbols('f C M')
        du, ru = symbols('du ru')

        bdf0 =  3 * 0.5/dt
        bdf1 =  -4 * 0.5/dt
        bdf2 =  1 * 0.5/dt

        v_n1 = bdf0 * u_n1 + bdf1 * u_n + bdf2 * u_nm1
        v_n = bdf0 * u_n + bdf1 * u_nm1 + bdf2 * u_nm2
        v_nm1 =  bdf0 * u_nm1 + bdf1 * u_nm2 + bdf2 * u_nm3

        a_n1 = bdf0 * v_n1 + bdf1 * v_n + bdf2 * v_nm1

        if self.numerical_scheme == 'Newton Raphson':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n1) * u_n1 )

        elif self.numerical_scheme == 'Picard':
            r_u = f - ( M * a_n1 + C * v_n1 + self.K(u_n) * u_n1 )
        
        print("ru = ", r_u)

        drudu = diff(r_u, u_n1)
        eq_u = ru + drudu * du
        sol = solve(eq_u, du)
        du = (sol[0])

        print("du = ", du )


    def bdf2_adaptive_time_step(self):
        # ### BDF2 ###
        # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
        print("##### BDF2 #####")
        u_n1, u_n, unm1, v_n1, v_n, vnm1, t, dt, M, C, f, bdf0, bdf1, bdf2 = symbols('u_n1 u_n u_nm1 v_n1 v_n v_nm1 t dt M C f, bdf0 bdf1 bdf2')
        du, dv, ru, rv = symbols('du dv ru rv')

        x_dot = bdf0 * u_n1 + bdf1 * u_n + bdf2 * unm1
        v_dot = bdf0 * v_n1 + bdf1 * v_n + bdf2 * vnm1

        r_u = - x_dot + v_n1

        if self.numerical_scheme == 'Newton Raphson':
            r_v = f - M * v_dot - self.K(u_n1) * u_n1 - C * v_n1

        elif self.numerical_scheme == 'Picard':
            r_v = f - M * v_dot - self.K(u_n) * u_n1 - C * v_n1

        print("ru = ", r_u)
        print("rv = ", r_v)

        drudu = diff(r_u, u_n1)
        drudv = diff(r_u, v_n1)
        drvdu = diff(r_v, u_n1)
        drvdv = diff(r_v, v_n1)

        eq1 = ru + (drudu * du + drudv * dv)
        eq2 = rv + (drvdu * du + drvdv * dv)

        sol = solve([eq1, eq2],(du, dv))
        du = sol[du]
        dv = sol[dv]

        print("du = ", du )
        print("dv = ", dv )


    def print_scheme(self):
        if self.time_scheme == 'euler_disp':
            self.euler_disp_based()

        if self.time_scheme == 'bdf1_disp':
            self.bdf1_disp_based()

        if self.time_scheme == 'bdf2_disp':
            self.bdf2_disp_based()
        
        if self.time_scheme == 'bdf2_adaptive':
            self.bdf2_adaptive_time_step()


if __name__ == "__main__":
    # Get command line arguments
    my_sdof = SDoF('bdf2_disp','Newton Raphson')
    #my_sdof = SDoF('bdf2_disp','Picard')
