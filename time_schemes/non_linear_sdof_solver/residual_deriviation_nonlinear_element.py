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


    def euler(self):
        # ### euler ###
        # v_n+1 = v_n + dt f(tn, v_n)
        print("##### euler #####")

        u_n1, u_n, v_n1, v_n, t, dt, f, M, C, K = symbols('u_n1 u_n v_n1 v_n t dt f M C K')
        du, dv, ru, rv = symbols('du dv ru rv')

        x_dot = (u_n1 - u_n) / dt
        v_dot = (v_n1 - v_n) / dt

        r_u = - x_dot + v_n
        r_v = f - M * v_dot - K * u_n - C * v_n

        print("ru = ", r_u)
        print("rv = ", r_v)

        drudu = diff(r_u, u_n1)
        drudv = diff(r_u, v_n1)
        drvdu = diff(r_v, u_n1)
        drvdv = diff(r_v, v_n1)

        eq1 = ru + (drudu * du + drudv * dv)
        eq2 = rv + (drvdu * du + drvdv * dv)

        sol = linsolve([eq1, eq2],(du, dv))
        du = sol.args[0][0]
        dv = sol.args[0][1]

        print("du = ", du )
        print("dv = ", dv )


    def bdf1(self):
        # ### BDF1 ###
        # v_n+1 = v_n + dt f(tn+1, v_n+1)
        print("##### BDF1 #####")

        u_n1, u_n, v_n1, v_n, t, dt, M, C, f = symbols('u_n1 u_n v_n1 v_n t dt M C f')
        du, dv, ru, rv = symbols('du dv ru rv')

        x_dot = (u_n1 - u_n) / dt
        v_dot = (v_n1 - v_n) / dt

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


    def bdf2(self):
        # ### BDF2 ###
        # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
        print("##### BDF2 #####")
        u_n1, u_n, unm1, v_n1, v_n, vnm1, t, dt, M, C, f = symbols('u_n1 u_n unm1 v_n1 v_n vnm1 t dt M C f')
        du, dv, ru, rv = symbols('du dv ru rv')

        c0 = 3 / ( 2 * dt)
        c1 = - 2 / dt
        c2 = 1 / ( 2 * dt)

        x_dot = c0 * u_n1 + c1 * u_n + c2 * unm1
        v_dot = c0 * v_n1 + c1 * v_n + c2 * vnm1

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


    def bdf2_adaptive_time_step(self):
        # ### BDF2 ###
        # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
        print("##### BDF2 #####")
        u_n1, u_n, unm1, v_n1, v_n, vnm1, t, dt, M, C, f, bdf0, bdf1, bdf2 = symbols('u_n1 u_n unm1 v_n1 v_n vnm1 t dt M C f, bdf0 bdf1 bdf2')
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


    def bdf1_adaptive_time_step_ave(self):
        print("##### BDF1 #####")
        u_bar_n1, u_bar_n, u_bar_nm1, v_bar_n1, v_bar_n, v_bar_nm1 = symbols('u_bar_n1 u_bar_n u_bar_nm1 v_bar_n1 v_bar_n v_bar_nm1')
        M, C, f, dt = symbols('M C f dt')
        du, dv, ru, rv = symbols('du dv ru rv')
        An1, Bn1, An, Bn = symbols('An1 Bn1 An Bn')

        u_n1 = (u_bar_n1 - An1 * u_bar_n) / Bn1
        v_n1 = (v_bar_n1 - An1 * v_bar_n) / Bn1

        u_n = (u_bar_n - An * u_bar_nm1) / Bn
        v_n = (v_bar_n - An * v_bar_nm1) / Bn        
        
        x_dot = (u_n1 - u_n) / dt
        v_dot = (v_n1 - v_n) / dt

        r_u = - x_dot + v_n1

        if self.numerical_scheme == 'Newton Raphson':
            r_v = f - M * v_dot - self.K(u_n1) * u_n1 - C * v_n1

        elif self.numerical_scheme == 'Picard':
            r_v = f - M * v_dot - self.K(u_n) * u_n1 - C * v_n1

        print("ru = ", r_u)
        print("rv = ", r_v)

        drudu = diff(r_u, u_bar_n1)
        drudv = diff(r_u, v_bar_n1)
        drvdu = diff(r_v, u_bar_n1)
        drvdv = diff(r_v, v_bar_n1)

        eq1 = ru + (drudu * du + drudv * dv)
        eq2 = rv + (drvdu * du + drvdv * dv)

        sol = solve([eq1, eq2],(du, dv))
        du = sol[du]
        dv = sol[dv]

        print("du = ", du )
        print("dv = ", dv )


    def bdf2_adaptive_time_step_ave(self):
        # ### BDF2 ###
        # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
        print("##### BDF2 #####")
        u_bar_n1, u_bar_n, u_bar_nm1, u_bar_nm2, v_bar_n1, v_bar_n, v_bar_nm1, v_bar_nm2 = symbols('u_bar_n1 u_bar_n u_bar_nm1 u_bar_nm2 v_bar_n1 v_bar_n v_bar_nm1 v_bar_nm2')
        M, C, f = symbols('M C f')
        bdf0, bdf1, bdf2 = symbols('bdf0 bdf1 bdf2')
        du, dv, ru, rv = symbols('du dv ru rv')
        An1, Bn1, An, Bn, Anm1, Bnm1 = symbols('An1 Bn1 An Bn Anm1 Bnm1')

        u_n1 = (u_bar_n1 - An1 * u_bar_n) / Bn1
        v_n1 = (v_bar_n1 - An1 * v_bar_n) / Bn1

        u_n = (u_bar_n - An * u_bar_nm1) / Bn
        v_n = (v_bar_n - An * v_bar_nm1) / Bn

        u_nm1 = (u_bar_nm1 - Anm1 * u_bar_nm2) / Bnm1
        v_nm1 = (v_bar_nm1 - Anm1 * v_bar_nm2) / Bnm1

        x_dot = bdf0 * u_n1 + bdf1 * u_n + bdf2 * u_nm1
        v_dot = bdf0 * v_n1 + bdf1 * v_n + bdf2 * v_nm1

        r_u = - x_dot + v_n1

        if self.numerical_scheme == 'Newton Raphson':
            r_v = f - M * v_dot - self.K(u_n1) * u_n1 - C * v_n1

        elif self.numerical_scheme == 'Picard':
            r_v = f - M * v_dot - self.K(u_n) * u_n1 - C * v_n1

        print("ru = ", r_u)
        print("rv = ", r_v)

        drudu = diff(r_u, u_bar_n1)
        drudv = diff(r_u, v_bar_n1)
        drvdu = diff(r_v, u_bar_n1)
        drvdv = diff(r_v, v_bar_n1)

        eq1 = ru + (drudu * du + drudv * dv)
        eq2 = rv + (drvdu * du + drvdv * dv)

        sol = solve([eq1, eq2],(du, dv))
        du = sol[du]
        dv = sol[dv]

        print("du = ", du )
        print("dv = ", dv )



    def rk4(self):
        # v_n+1 = v_n + f( (k0 + k1 + k2 + k3) / 6 ) * dt
        u_n1, u_n, v_n1, v_n, t, dt, M, C, f, K = symbols('u_n1 u_n v_n1 v_n t dt M C f K')
        ru, rv, ru0, rv0, ru1, rv1, ru2, rv2, ru3, rv3 = symbols('ru rv ru0 rv0 ru1 rv1 ru2 rv2 ru3 rv3')
        du, dv = symbols('du dv')
        print("##### RK4 #####")

        _ru0 = self.g(v_n)
        _rv0 = self.rhs(t, u_n, v_n)

        _ru1 = self.g(v_n + 0.5*rv0*dt)
        _rv1 = self.rhs(t + 0.5*dt, u_n + 0.5*ru0*dt, v_n + 0.5*rv0*dt)

        _ru2 = self.g(v_n + 0.5*rv1*dt)
        _rv2 = self.rhs(t + 0.5*dt, u_n + 0.5*ru1*dt, v_n + 0.5*rv1*dt)

        _ru3 = self.g(v_n + rv2*dt)
        _rv3 = self.rhs(t + dt, u_n + ru2*dt, v_n + rv2*dt)

        print("ru0 = ", _ru0)
        print("rv0 = ", _rv0)
        print("ru1 = ", _ru1)
        print("rv1 = ", _rv1)
        print("ru2 = ", _ru2)
        print("rv2 = ", _rv2)
        print("ru3 = ", _ru3)
        print("rv3 = ", _rv3)

        r_u = nsimplify( (u_n1 - u_n)/dt - 1/6 * (ru0 + 2*ru1 + 2*ru2 + ru3) )
        r_v = nsimplify( (v_n1 - v_n)/dt - 1/6 * (rv0 + 2*rv1 + 2*rv2 + rv3) )

        print("ru = ", r_u)
        print("rv = ", r_v)

        drudu = diff(r_u, u_n1)
        drudv = diff(r_u, v_n1)
        drvdu = diff(r_v, u_n1)
        drvdv = diff(r_v, v_n1)
        print("drudu: ", drudu)
        print("drudv: ", drudv)
        print("drvdu: ", drvdu)
        print("drvdv: ", drvdv)

        eq1 = ru + (drudu * du + drudv * dv)
        eq2 = rv + (drvdu * du + drvdv * dv)
        print("eq1: ", eq1)
        print("eq2: ", eq2)

        sol = linsolve([eq1, eq2],(du, dv))
        du = sol.args[0][0]
        dv = sol.args[0][1]

        print("du = ", du )
        print("dv = ", dv )


    def print_scheme(self):
        if self.time_scheme == 'euler':
            self.euler()

        if self.time_scheme == 'bdf1':
            self.bdf1()

        if self.time_scheme == 'bdf1_adaptive_time_step_ave':
            self.bdf1_adaptive_time_step_ave()

        if self.time_scheme == 'bdf2':
            self.bdf2()

        if self.time_scheme == 'rk4':
            self.rk4()
        
        if self.time_scheme == 'bdf2_adaptive_time_step':
            self.bdf2_adaptive_time_step()
        
        if self.time_scheme == 'bdf2_adaptive_time_step_ave':
            self.bdf2_adaptive_time_step_ave()


if __name__ == "__main__":
    # Get command line arguments
    my_sdof = SDoF('bdf2_adaptive_time_step_ave','Newton Raphson')
