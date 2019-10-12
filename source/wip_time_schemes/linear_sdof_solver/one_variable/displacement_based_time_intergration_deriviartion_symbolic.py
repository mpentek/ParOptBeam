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

an1, an, un2, un1, un, unm1, unm2, unm3, vn1, vn, vnm1, vnm2, t, dt = symbols('an1 an un2 un1 un unm1 unm2 unm3 vn1 vn vnm1 vnm2 t SDoF.dt')
f, K, C, M = symbols('SDoF.f(t) SDoF.K SDoF.C SDoF.M')

time_scheme = None

def euler_disp_based():
    # ### EULER ###
    # M * u''(t) + C * u'(t) + K * u(t) = f)
    global an, un2, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    vn = (un1 - un) / dt
    vnm1 = ( un - unm1) / dt
    vnm2 = ( unm1 - unm2) / dt
    anm1 = ( vn - vnm1 ) / dt

    eq_u = M * anm1 + C * vnm1 + K * unm1 - f
    #print(eq_u)

    sol = solve(eq_u, un1)
    un1 = sol[0]

    print("##### EULER #####")
    print("u_n1 = ", un1)

def bdf1_disp_based():
    # ### BDF1 ###
    # M * u''(t) + C * u'(t) + K * u(t) = f
    global an, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M

    vn1 = ( un1 - un) / dt
    vn = ( un - unm1) / dt
    an1 = ( vn1 - vn ) / dt

    eq_u = M * an1 + C * vn1 + K * un1 - f
    #print(eq_u)

    sol = solve(eq_u, un1)
    un1 = (sol[0])

    print("##### BDF1 #####")
    print("u_n1 = ", un1)


def bdf2_disp_based():
    # ### BDF2 ###
    # un+1 = 4/3 un - 1/3 un-1 + 2/3 dt f(tn+1, un+1)
    # vn+1 = 0.5/dt * (3 un+1 - 4 un + unm1)
    global an1, an, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M
    bdf0, bdf1, bdf2 = symbols('bdf0, bdf1, bdf2')
    # bdf0 =  3 * 0.5/dt
    # bdf1 =  -4 * 0.5/dt
    # bdf2 =  1 * 0.5/dt

    vn1 = bdf0 * un1 + bdf1 * un + bdf2 * unm1
    vn = bdf0 * un + bdf1 * unm1 + bdf2 * unm2
    vnm1 = bdf0 * unm1 + bdf1 * unm2 + bdf2 * unm3

    an1 = bdf0 * vn1 + bdf1 * vn + bdf2 * vnm1

    eq_u = nsimplify (M * an1 + C * vn1 + K * un1 - f)
    #print(eq_u)

    sol = solve(eq_u, un1)
    un1 = nsimplify (sol[0])

    print("##### BDF2 #####")
    print("u_n1 = ", un1)

    return un1


def bdf2_disp_based_adaptive():
    # ### BDF2 ###
    # un+1 = 4/3 un - 1/3 un-1 + 2/3 dt f(tn+1, un+1)
    # vn+1 = 0.5/dt * (3 un+1 - 4 un + unm1)
    global an1, an, un1, un, unm1, unm2, vn1, vn, vnm1, vnm2, t, dt, f, K, C, M
    un1, un, unm1, vn1, vn, vnm1, bdf0, bdf1, bdf2, t = symbols('un1 un unm1 vn1 vn vnm1 bdf0 bdf1 bdf2 t')

    vn1 = bdf0 * un1 + bdf1 * un + bdf2 * unm1
    vn = bdf0 * un + bdf1 * unm1 + bdf2 * unm2
    vnm1 =  bdf0 * unm1 + bdf1 * unm2 + bdf2 * unm3

    an1 = bdf0 * vn1 + bdf1 * vn + bdf2 * vnm1

    eq_u = nsimplify (M * an1 + C * vn1 + K * un1 - f)
    #print(eq_u)

    sol = solve(eq_u, un1)
    un1 = nsimplify (sol[0])

    print("##### BDF2 #####")
    print("u_n1 = ", un1)

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
        euler_disp_based()

    if time_scheme == 'bdf1':
        bdf1_disp_based()

    if time_scheme == 'bdf2':
        bdf2_disp_based()
    
    if time_scheme == 'bdf2_adaptive':
        bdf2_disp_based_adaptive()

    if time_scheme == 'rk4':
        rk4()


# Check number of command line arguments
# if len(sys.argv) != 2:
#     print ("Usage: python derive_scheme.py <scheme>")
#     sys.exit(1)

if __name__ == "__main__":
    # Get command line arguments
    # time_scheme = sys.argv[1]
    print_scheme('rk4')
