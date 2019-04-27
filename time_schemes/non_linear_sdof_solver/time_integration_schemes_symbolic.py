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

    def __init__(self, scheme=None, K=1.0, M=1, C=0.1, f=0.1, u0=1.0, v0=0.0, dt=0.1):
        self.K = lambda u: (u**2 + 1)
        self.M = M
        self.C = C
        self.f = f
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.scheme = scheme
        self.print_scheme()


    def rhs(self, t, u, v):
        f, K, C, M = symbols('f K C M')
        return (f - K * u - C * v) / M


    # Function for the u'(t) = v ODE.
    def g(self, v):
        return v


    def euler(self):
        # ### euler ###
        # vn+1 = vn + dt f(tn, vn)
        un1, un, vn1, vn, t, dt = symbols('un1 un vn1 vn t dt')

        f_v = self.rhs(t, un, vn)
        f_u = self.g(vn)

        eq_v = vn1 - ( vn + f_v * dt )
        eq_u = un1 - ( un + f_u * dt )

        print(eq_v)
        print(eq_u)

        sol = linsolve([eq_v, eq_u],(vn1, un1))
        vn1 = sol.args[0][0]
        un1 = sol.args[0][1]

        print("##### euler #####")
        print("un1 = ", un1)
        print("vn1 = ", vn1)


    def bdf1(self):
        # ### BDF1 ###
        # vn+1 = vn + dt f(tn+1, vn+1)
        un1, un, vn1, vn, t, dt, K, M, C, f = symbols('un1 un vn1 vn t dt K M C f')

        f_v = self.rhs(t, un1, vn1)
        f_u = self.g(vn1)

        eq_v = vn1 - ( vn + f_v * dt )
        eq_u = un1 - ( un + f_u * dt )

        sol = solve([eq_v, eq_u],(vn1, un1))
        un1 = sol[un1]
        vn1 = sol[vn1]

        print("##### BDF1 #####")
        print("un1 = ", un1)
        print("vn1 = ", vn1)


    def bdf2(self):
        # ### BDF2 ###
        # vn+1 = 4/3 vn - 1/3 vn-1 + 2/3 dt f(tn+1, vn+1)
        un1, un, unm1, vn1, vn, vnm1, t, dt = symbols('un1 un unm1 vn1 vn vnm1 t dt')

        f_v = self.rhs(t, un1, vn1)
        f_u = self.g(vn1)

        eq_v = vn1 - ( 4/3 * vn - 1/3 * vnm1 + 2/3 * dt * f_v )
        eq_u = un1 - ( 4/3 * un - 1/3 * unm1 + 2/3 * dt * f_u )

        sol = solve([eq_v, eq_u],(vn1, un1))
        un1 = sol[un1]
        vn1 = sol[vn1]

        print("##### BDF2 #####")
        print("un1 = ", un1)
        print("vn1 = ", vn1)

        return un1, vn1


    def bdf2_adaptive_time_step(self):
        # ### BDF2 ###
        # vn+1 - 4/3 vn + 1/3 vn-1 = 2/3 dt f(tn+1, vn+1)
        # bdf0 * vn+1 + bdf1*vn + bdf2 * vnm1 = f(tn+1, vn+1)
        un1, un, unm1, vn1, vn, vnm1, bdf0, bdf1, bdf2, t = symbols('u_n1 u_n u_nm1 v_n1 v_n v_nm1 bdf0 bdf1 bdf2 t')

        f_v = self.rhs(t, un1, vn1)
        f_u = self.g(vn1)

        print(f_v)

        eq_v = bdf0 * vn1 + bdf1*vn + bdf2 * vnm1 - f_v
        eq_u = bdf0 * un1 + bdf1*un + bdf2 * unm1 - f_u

        sol = solve([eq_v, eq_u],(vn1, un1))
        un1 = sol[un1]
        vn1 = sol[vn1]

        print("##### BDF2 #####")
        print("un1 = ", un1)
        print("vn1 = ", vn1)

        return un1, vn1


    def bdf1_adaptive_time_step_ave(self):
        # ### BDF1 ###
        # vn+1 = vn + dt f(tn+1, vn+1)
        u_bar_n1, u_bar_n, u_bar_nm1, v_bar_n1, v_bar_n, v_bar_nm1 = symbols('u_bar_n1 u_bar_n u_bar_nm1 v_bar_n1 v_bar_n v_bar_nm1')
        M, C, f, dt , t = symbols('M C f dt t')
        An1, Bn1, An, Bn = symbols('An1 Bn1 An Bn')

        u_n1 = (u_bar_n1 - An1 * u_bar_n) / Bn1
        v_n1 = (v_bar_n1 - An1 * v_bar_n) / Bn1

        u_n = (u_bar_n - An * u_bar_nm1) / Bn
        v_n = (v_bar_n - An * v_bar_nm1) / Bn    

        f_v = self.rhs(t, u_n1, v_n1)
        f_u = self.g(v_n1)

        eq_v = v_n1 - ( v_n + f_v * dt )
        eq_u = u_n1 - ( u_n + f_u * dt )

        sol = solve([eq_v, eq_u],(v_bar_n1, u_bar_n1))
        u_bar_n1 = sol[u_bar_n1]
        v_bar_n1 = sol[v_bar_n1]

        print("##### BDF1 #####")
        print("u_bar_n1 = ", u_bar_n1)
        print("v_bar_n1 = ", v_bar_n1)


    def bdf2_adaptive_time_step_ave(self):
        # ### BDF2 ###
        # vn+1 - 4/3 vn + 1/3 vn-1 = 2/3 dt f(tn+1, vn+1)
        # bdf0 * vn+1 + bdf1*vn + bdf2 * vnm1 = f(tn+1, vn+1)
        
        u_bar_n1, u_bar_n, u_bar_nm1, u_bar_nm2, v_bar_n1, v_bar_n, v_bar_nm1, v_bar_nm2 = symbols('u_bar_n1 u_bar_n u_bar_nm1 u_bar_nm2 v_bar_n1 v_bar_n v_bar_nm1 v_bar_nm2')
        M, C, f, t = symbols('M C f t')
        bdf0, bdf1, bdf2 = symbols('bdf0 bdf1 bdf2')
        An1, Bn1, An, Bn, Anm1, Bnm1 = symbols('An1 Bn1 An Bn Anm1 Bnm1')

        u_n1 = (u_bar_n1 - An1 * u_bar_n) / Bn1
        v_n1 = (v_bar_n1 - An1 * v_bar_n) / Bn1

        u_n = (u_bar_n - An * u_bar_nm1) / Bn
        v_n = (v_bar_n - An * v_bar_nm1) / Bn

        u_nm1 = (u_bar_nm1 - Anm1 * u_bar_nm2) / Bnm1
        v_nm1 = (v_bar_nm1 - Anm1 * v_bar_nm2) / Bnm1
        
        f_v = self.rhs(t, u_n1, v_n1)
        f_u = self.g(v_n1)

        eq_v = bdf0 * v_n1 + bdf1*v_n + bdf2 * v_nm1 - f_v
        eq_u = bdf0 * u_n1 + bdf1*u_n + bdf2 * u_nm1 - f_u

        sol = solve([eq_v, eq_u],(v_bar_n1, u_bar_n1))
        u_bar_n1 = sol[u_bar_n1]
        v_bar_n1 = sol[v_bar_n1]

        print("##### BDF2 #####")
        print("u_bar_n1 = ", u_bar_n1)
        print("v_bar_n1 = ", v_bar_n1)

        return u_bar_n1, v_bar_n1


    def rk4(self):
        # vn+1 = vn + f( (k0 + k1 + k2 + k3) / 6 ) * dt
        un1, un, vn1, vn, t, dt = symbols('un1 un vn1 vn t dt')
        k0, k1, k2, k3, l0, l1, l2, l3 = symbols ('k0 k1 k2 k3 l0 l1 l2 l3')

        _k0 = dt*self.g(vn)
        _l0 = dt*self.rhs(t, un, vn)

        print("k0 = ", _k0)
        print("l0 = ", _l0)

        _k1 = dt*self.g(vn + 0.5*l0)
        _l1 = dt*self.rhs(t + 0.5*dt, un + 0.5*k0, vn + 0.5*l0)

        print("k1 = ", _k1)
        print("l1 = ", _l1)

        _k2 = dt*self.g(vn + 0.5*l1)
        _l2 = dt*self.rhs(t + 0.5*dt, un + 0.5*k1, vn + 0.5*l1)

        print("k2 = ", _k2)
        print("l2 = ", _l2)

        _k3 = dt*self.g(vn + l2)
        _l3 = dt*self.rhs(t + dt, un + k2, vn + l2)

        print("k3 = ", _k3)
        print("l3 = ", _l3)

        un1 = nsimplify (un + (1.0/6.0) * (k0 + 2*(k1+k2) +k3) )
        vn1 = nsimplify (vn + (1.0/6.0) * (l0 + 2*(l1+l2) +l3) )

        print("##### RK4 #####")
        print("un1 = ", un1)
        print("vn1 = ", vn1)

        return un1, vn1

    def print_scheme(self):
        if self.scheme == 'euler':
            self.euler()

        if self.scheme == 'bdf1':
            self.bdf1()

        if self.scheme == 'bdf2':
            self.bdf2()

        if self.scheme == 'rk4':
            self.rk4()
        
        if self.scheme == 'bdf2_adaptive_time_step':
            self.bdf2_adaptive_time_step()
        
        if self.scheme == 'bdf1_adaptive_time_step_ave':
            self.bdf1_adaptive_time_step_ave()

        if self.scheme == 'bdf2_adaptive_time_step_ave':
            self.bdf2_adaptive_time_step_ave()

# Check number of command line arguments
if len(sys.argv) != 2:
	print ("Usage: python derive_scheme.py <scheme>")
	sys.exit(1)

if __name__ == "__main__":
    # Get command line arguments
    scheme = sys.argv[1]
    my_sdof = SDoF(scheme)
