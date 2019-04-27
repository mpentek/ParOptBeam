import cmath
import math
import numpy as np

def analytical_general(SDoF, t):
	# !!! This analytical solution only works for constant f !!!
	if ( (SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M) > 0):
			Delta = math.sqrt( SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M )
	else:
			Delta = cmath.sqrt( SDoF.C * SDoF.C - 4 * SDoF.K * SDoF.M )

	A1 = - ( - SDoF.C * SDoF.f(t) + SDoF.u0 * SDoF.C * SDoF.K + 2 * SDoF.v0 * SDoF.K * SDoF.M \
			+ SDoF.f(t) * Delta - SDoF.u0 * SDoF.K * Delta ) / (2 * SDoF.K * Delta)
	A2 = - ( SDoF.C * SDoF.f(t) - SDoF.u0 * SDoF.C * SDoF.K - 2 * SDoF.v0 * SDoF.K * SDoF.M \
			+ SDoF.f(t) * Delta - SDoF.u0 * SDoF.K * Delta ) /( 2 * SDoF.K * Delta)
	# v0 = 1 / ( 2 * SDoF.M) * ( A2 * ( Delta - A3 ) - A1 * ( Delta + A3 ) )
	A3 = ( A2 * Delta - A1 * Delta  - 2 * SDoF.v0 * SDoF.M ) / ( A1 + A2 )

	# source: Wolfram Alpha
	u_n1 = A1 * cmath.exp( 0.5 * t * (- Delta / SDoF.M  - SDoF.C / SDoF.M )) + \
			A2 * cmath.exp( 0.5 * t * ( Delta / SDoF.M  - SDoF.C / SDoF.M )) + SDoF.f(t) / SDoF.K

	v_n1 = 1 / ( 2 * SDoF.M) * cmath.exp( -  ( t *  ( Delta + A3 ) ) / ( 2 * SDoF.M ) ) * \
					( A2 * ( Delta - A3 ) * cmath.exp( t * Delta / SDoF.M ) - A1 * ( Delta + A3 ) )

	return u_n1, v_n1


# Analytical solutions for perticular cases
def analytical(SDoF, t):
	wn = math.sqrt( SDoF.K / SDoF.M )
	xi = SDoF.C / (2 * math.sqrt ( SDoF.K * SDoF.M) )
	#print('xi is', xi)

	A = math.sqrt( SDoF.u0 * SDoF.u0 + (SDoF.v0 / wn) * (SDoF.v0 / wn) )
	phi = np.arctan( SDoF.v0 / (SDoF.u0 * wn) )

	c_crit = 2 * SDoF.M * wn
	D = SDoF.C / c_crit

	if (xi < 1): wd = wn * math.sqrt ( 1 - xi * xi )
	if (xi > 1): D1 = math.sqrt ( D * D - 1 )

	# Analytical solutions for perticular cases
	if (SDoF.f == 0):
		# Simple harmonic oscillator without damping for u(0) = 1, v(0) = 0
		if (SDoF.C == 0):
				u_n1 = A * math.cos ( wn*t + phi)
		# Harmonic oscillator with damping for u(0) = 1, v(0) = 0
		else:
			# Underdamped
			if (xi < 1):
				u_n1 = A * math.exp ( - D * wn * t ) * math.cos ( wd * t - phi)
			# Critically overdamped
			# SWE Lecture 3 Structual Anaylsis
			elif (xi > 1):
				u_n1 = ((SDoF.u0 * wn * (D + D1) + SDoF.v0) / (2 * wn * D1 )) * math.exp( (-D + D1) * wn * t ) - \
						((SDoF.u0 * wn * (D - D1) + SDoF.v0) / (2 * wn * D1 )) * math.exp( (-D - D1 )* wn * t )
			# Critically damped
			else:
				u_n1 = A * math.exp ( wn * t ) * ( SDoF.u0 + ( SDoF.v0 + wn * SDoF.u0 ) * t )
	else:
		u_n1 = analytical_general(SDoF, t)

	return u_n1


def euler(SDoF, t, un, vn):

    u_n1 = SDoF.dt * vn + un
    v_n1 = ( SDoF.M * vn - SDoF.dt * (SDoF.C * vn + SDoF.K * un - SDoF.f(t)) ) / SDoF.M

    return u_n1, v_n1


def bdf1(SDoF, t, un, vn):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K
    f = SDoF.f(t)
    dt = SDoF.dt

    u_n1 = (dt*(M*vn + dt*f) + un*(C*dt + M))/(C*dt + K*dt**2 + M)
    v_n1 = (-K*dt*un + M*vn + dt*f)/(C*dt + K*dt**2 + M)

    return u_n1, v_n1


def bdf2(SDoF, t, un, vn, unm1, vnm1):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K
    f = SDoF.f(t)
    dt = SDoF.dt

    u_n1 = (2.0*dt*(4.0*M*vn - M*vnm1 + 2.0*dt*f) + (4.0*un - unm1)*(2.0*C*dt + 3.0*M))/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)
    v_n1 = (-8.0*K*dt*un + 2.0*K*dt*unm1 + 12.0*M*vn - 3.0*M*vnm1 + 6.0*dt*f)/(6.0*C*dt + 4.0*K*dt**2 + 9.0*M)

    return u_n1, v_n1


def rk4(SDoF, t, un, vn):

    C = SDoF.C
    M = SDoF.M
    K = SDoF.K
    f = SDoF.f(t)
    dt = SDoF.dt

    k0 =  dt*vn
    l0 =  dt*(-C*vn - K*un + f)/M
    k1 =  dt*(0.5*l0 + vn)
    l1 =  dt*(-C*(0.5*l0 + vn) - K*(0.5*k0 + un) + f)/M
    k2 =  dt*(0.5*l1 + vn)
    l2 =  dt*(-C*(0.5*l1 + vn) - K*(0.5*k1 + un) + f)/M
    k3 =  dt*(l2 + vn)
    l3 =  dt*(-C*(l2 + vn) - K*(k2 + un) + f)/M
    u_n1 =  k0/6 + k1/3 + k2/3 + k3/6 + un
    v_n1 =  l0/6 + l1/3 + l2/3 + l3/6 + vn

    return u_n1, v_n1