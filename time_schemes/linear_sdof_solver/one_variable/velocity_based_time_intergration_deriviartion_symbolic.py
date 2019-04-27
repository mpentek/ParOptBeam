################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import cmath
import sys

init_printing(use_unicode=True)

an1, an, un2, un1, un, unm1, unm2, unm3, vn2, vn1, vn, vnm1, vnm2, t, dt = symbols('an1 an un2 un1 un unm1 unm2 unm3 vn2 vn1 vn vnm1 vnm2 t SDoF.dt')
f, K, C, M = symbols('SDoF.f(t) SDoF.K SDoF.C SDoF.M')

time_scheme = None

def euler():
    # ### euler ###
    # M * u''(t) + C * u'(t) + K * u(t) = f
    global un1, unm1, unm2, vn2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    # It is impossible to express vn1 with pure velocity components as velocity need to be
    # integrated to describe the displacemt in M * u''(t) + C * u'(t) + K * u(t) = f
    # unm1 = unm2 + vnm2 * dt | -> this will be an infinetive substraction till u0
    # un = unm1 + vnm1 * dt

    an = (vn1 - vn) / dt
    eq_v = M * an + C * vn + K * un - f
    print(eq_v)

    sol = solve(eq_v, vn1)
    vn1 = sol[0]

    # **********************************************************************
    # The following code was tried, but the result delivered was not correct
    # Please call "euler2" in sdof_solver.py to check its result
    # unm1 = unm2 + vnm2 * dt
    # un = unm1 + vnm1 * dt
    # an = (vn1 - vn) / dt

    # eq_v = M * an + C * vn + K * un - f
    # print(eq_v)

    # sol = solve(eq_v, unm1)
    # unm1 = sol[0]
    # print("u_nm1 = ", unm1)

    # _unm2 = unm1.subs({vn1: vn, vn: vnm1, vnm1: vnm2})
    # print("u_nm2 = ", _unm2)

    # sol = solve(eq_v, vn1)
    # vn1 = sol[0]
    # vn1 = vn1.subs({unm2: _unm2})

    print("##### euler #####")
    print("v_n1 = ", vn1)


def bdf1():
    # ### bdf1 ###
    # M * u''(t) + C * u'(t) + K * u(t) = f
    global an, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    un1 = un + vn1 * dt
    an1 = (vn1 - vn) / dt

    eq_u = M * an1 + C * vn1 + K * un1 - f
    #print(eq_u)

    sol = solve(eq_u, un1)
    un1 = sol[0]
    _un = un1.subs({vn1: vn, vn: vnm1})
    #print("u_n = ", _un)

    sol = solve(eq_u, vn1)
    vn1 = sol[0]
    vn1 = vn1.subs({un: _un})
    #print(eq_u)

    print("##### bdf1 #####")
    print("v_n1 = ", vn1)


def bdf2():
    # ### BDF2 ###
    # un+1 = 4/3 un - 1/3 un-1 + 2/3 dt f(tn+1, un+1)
    # vn+1 = 0.5/dt * (3 un+1 - 4 un + unm1)
    global an1, an, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    c0 =  3 * 0.5/dt
    c1 =  -4 * 0.5/dt
    c2 =  1 * 0.5/dt

    un1 = 4/3 * un - 1/3 * unm1 + 2/3 * dt * vn1
    an1 = c0 * vn1 + c1 * vn + c2 * vnm1

    eq_u = nsimplify (M * an1 + C * vn1 + K * un1 - f)
    print(eq_u)

    sol = solve(eq_u, vn1)
    vn1 = nsimplify (sol[0])

    print("##### BDF2 #####")
    print("v_n1 = ", vn1)

    return un1


def rk4():
    # un+1 = un + g( (k0 + k1 + k2 + k3) / 6 ) * dt
    # vn+1 = vn + rhs( (l0 + l1 + l2 + l3) / 6 ) * dt
    global an1, an, un2, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    k0 = dt*vn
    k1 = dt*(vn + 0.5*k0)
    k2 = dt*(vn + 0.5*k1)
    k3 = dt*(vn + k2)

    eq_u = nsimplify (un1 - (un + (k0 + k1 + k2 + k3) / 6))
    sol = solve(eq_u, vn)
    vn = nsimplify (sol[0])
    print("vn = ", vn)

    l0 =  dt*(-C*vn - K*un + f)/M
    l1 =  dt*(-C*(0.5*l0 + vn) - K*(0.5*k0 + un) + f)/M
    l2 =  dt*(-C*(0.5*l1 + vn) - K*(0.5*k1 + un) + f)/M
    l3 =  dt*(-C*(l2 + vn) - K*(k2 + un) + f)/M

    eq_v = nsimplify (vn1 - (vn + (l0 + l1 + l2 + l3) / 6))
    sol = solve(eq_v, vn1)
    vn1 = nsimplify (sol[0])

    print("vn1 = ", vn1)

    an = (l0 + l1 + l2 + l3) / 6

    eq_a = an - (f - K * un - C * vn) / M
    print("eq_a: ", eq_a)
    sol = solve(eq_a, un1)
    un1 = nsimplify (sol[0])
    un1.subs({vn: 24*(-un + un1)/(dt*(dt**3 + 3*dt**2 + 8*dt + 16))})

    print("##### RK4 #####")
    print("un1 = ", un1)

    return un1


def print_scheme(time_scheme):
    if time_scheme == 'euler':
        euler()

    if time_scheme == 'bdf1':
        bdf1()

    if time_scheme == 'bdf2':
        bdf2()

    if time_scheme == 'rk4':
        rk4()


# Check number of command line arguments
if len(sys.argv) != 2:
    print ("Usage: python derive_scheme.py <scheme>")
    sys.exit(1)

if __name__ == "__main__":
    # Get command line arguments
    time_scheme = sys.argv[1]
    print_scheme(time_scheme)
