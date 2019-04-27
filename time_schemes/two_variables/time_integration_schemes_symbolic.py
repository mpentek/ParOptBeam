################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
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
        self.K = K
        self.M = M
        self.C = C
        self.f = f
        self.u0 = u0
        self.v0 = v0
        self.dt = dt
        self.scheme = scheme
        self.wn = float( sqrt( self.K / self.M ) )
        self.T = float( 2 * np.pi / self.wn )
        self.tend = float( 2 * self.T )
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


# Check number of command line arguments
if len(sys.argv) != 2:
	print ("Usage: python derive_scheme.py <scheme>")
	sys.exit(1)

if __name__ == "__main__":
    # Get command line arguments
    scheme = sys.argv[1]
    my_sdof = SDoF(scheme)
