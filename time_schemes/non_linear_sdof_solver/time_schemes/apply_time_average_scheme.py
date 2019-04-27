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
import sys

init_printing(use_unicode=True)

def solve_averaged():
    u_ave_n1, u_ave_n, u_ave_nm1, v_ave_n1, v_ave_n, v_ave_nm1, t, dt = symbols('u_ave_n1 u_ave_n u_ave_nm1 v_ave_n1 v_ave_n v_ave_nm1 t dt')

    n = symbols('n')

    A = lambda n : 1 / n
    B = lambda n : (n - 1) / n

    u_n = (u_ave_n - B(n) * u_ave_nm1) / A(n)
    v_n = (v_ave_n - B(n) * v_ave_nm1) / A(n)

    u_n1, v_n1 = symbols('u_n1 v_n1')

    u_n1 = ( u_ave_n1 - B(n+1) * u_ave_n ) / A(n+1)
    v_n1 = ( v_ave_n1 - B(n+1) * v_ave_n ) / A(n+1)

    print("u_n1: ", u_n1)

if __name__ == "__main__":
    solve_averaged()