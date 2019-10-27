"""
################################################################################################
###   M * u''(t) + C * u'(t) + K * u(t) = f rewrite 2nd order ODE into system of 1st order ODEs
###   (I)  v'(t) = ( f - C * v(t) - K * u(t) ) / M = f(t, u, v) = rhs
###   (II) u'(t) = v(t)
###   differential equations in the form (d^2)y/(du^2) = (rhs)
###   rhs: The right-hand side function of the ODE.
###   Newton's 2nd Law formalism has been kept (rhs = f(t, u, v)/m)
###   rhs = f - cv - ku
################################################################################################
"""
from sympy import *

init_printing(use_unicode=True)

a_n1, a_n, u_n2, u_n1, u_n, u_nm1, u_nm2, u_nm3, t, dt = symbols(
    'a1 an u2 u1 self.un1 self.un2 self.un3 self.un4 t dt')

f, C, M, K = symbols('f self.B self.M self.K')


def euler():
    # ### euler ###
    # v_n+1 = v_n + dt f(tn, v_n)
    print("##### Euler #####")

    v_n = (u_n1 - u_n) / dt
    v_nm1 = (u_n - u_nm1) / dt
    a_nm1 = (v_n - v_nm1) / dt

    du, ru = symbols('du ru')

    r_u = f - (M * a_nm1 + C * v_nm1 + K * u_nm1)
    print("ru = ", r_u)

    drudu = diff(r_u, u_n1)
    eq_u = ru + drudu * du
    sol = solve(eq_u, du)
    du = (sol[0])

    print("du = ", du)


def bdf1():
    # ### BDF1 ###
    # v_n+1 = v_n + dt f(tn+1, v_n+1)
    print("##### BDF1 #####")

    v_n1 = (u_n1 - u_n) / dt
    v_n = (u_n - u_nm1) / dt
    a_n1 = (v_n1 - v_n) / dt

    du, ru = symbols('du ru')

    r_u = f - (M * a_n1 + C * v_n1 + K * u_n1)

    print("ru = ", r_u)

    drudu = diff(r_u, u_n1)
    eq_u = ru + drudu * du
    sol = solve(eq_u, du)
    du = (sol[0])

    print("du = ", du)


def bdf2():
    # ### BDF2 ###
    # v_n+1 = 4/3 v_n - 1/3 v_n-1 + 2/3 dt f(tn+1, v_n+1)
    print("##### BDF2 #####")

    bdf0, bdf1, bdf2 = symbols('self.bdf0 self.bdf1 self.bdf2')

    v_n1 = bdf0 * u_n1 + bdf1 * u_n + bdf2 * u_nm1
    v_n = bdf0 * u_n + bdf1 * u_nm1 + bdf2 * u_nm2
    v_nm1 = bdf0 * u_nm1 + bdf1 * u_nm2 + bdf2 * u_nm3

    a_n1 = bdf0 * v_n1 + bdf1 * v_n + bdf2 * v_nm1

    du, ru = symbols('du ru')

    r_u = f - (M * a_n1 + C * v_n1 + K * u_n1)

    print("ru = ", r_u)

    drudu = diff(r_u, u_n1)
    eq_u = ru + drudu * du
    sol = solve(eq_u, du)
    du = (sol[0])

    print("du = ", du)


if __name__ == "__main__":
    euler()
    bdf1()
    bdf2()
