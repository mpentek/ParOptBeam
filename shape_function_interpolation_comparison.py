import numpy as np
from math import radians, tan, cos, pi

# first set of shape functions

def interpolate_points_1(v, x, y, dydx):
    '''
    3rd order polynomial for a beam with point load
    in a generic form
    '''
    a = 0
    b = 0
    [c, d] = np.linalg.solve([[x[1]**2, x[1]**3],[2*x[1], 3*x[1]**2]], [y[1], dydx[1]])
    return [(a + b*s + c*s**2 + d*s**3) for s in v]

def interpolate_points_3(v, x, y, zeta=2):
    '''
    exponential interpolation
    from Ec F3 - F13
    valoue could also be 1.5-2.5
    '''
    a = y[1]
    return [a*(s/x[1])**zeta for s in v]

def interpolate_points_2(v, x, y):
    '''
    cosine assumption
    seen as a simplification for mode shapes
    first mode for cantilever beind 1/4 period of cos
    '''
    a = y[1] / (1 - cos(pi/2 * x[1] / v[-1])) 
    return [a * (1-cos(pi/2 * s/v[-1])) for s in v]

def interpolate_points_4(v, x, y):
    '''
    based upon the generic definition for mode shapes
    '''
    rho = 160
    area = 45 * 30
    rho_l = rho * area
    L = v[-1]
    f = 0.2
    w = 2 * f * np.pi

    EI = (f * 2 * np.pi * L**2 / 3.5156)**2 * rho_l

    # 3.51 = lam**2 = 1.875**2

    lam = ((w**2*rho_l*L**4)/EI)**0.25
    lam_2 = lam/L
    lam_3 = lam_2*x[1]

    a = np.array([[1, 0, 1, 0],
                  [0, lam_2, 0, lam_2],
                  [np.cosh(lam_3), np.sinh(lam_3), cos(lam_3), np.sin(lam_3)],
                  [np.sinh(lam_3), np.cosh(lam_3), -np.sin(lam_3), np.cos(lam_3)]])
    b = np.array([0, 0, y[1], dydx[1]/lam_2])
    c = np.linalg.solve(a, b)

    # a_red = np.array([[np.cosh(lam_3) - np.cos(lam_3), np.sinh(lam_3) - np.sin(lam_3)],
    #               [np.sinh(lam_3) + np.sin(lam_3), np.cosh(lam_3) - np.cos(lam_3)]])
    # b_red = np.array([y[1],dydx[1]/lam_2])
    # c_red = np.linalg.solve(b_red, a_red)

    # c_2 = [c_red[0], c_red[1], -c_red[0], -c_red[1]]

    return [c[0]*np.cosh(lam_2*s) + c[1]*np.sinh(lam_2*s) + c[2]
     * np.cos(lam_2*s) + c[3]*np.sin(lam_2*s) for s in v]

# secind set of shape functions

def interpolate_points_5(v, x, y, dydx):
    '''
    modified exponential
    to include the effect of rotation
    '''
    b = x[1] * dydx[1] / y[1]
    a = y[1] / (x[1]/v[-1])**b
    return [a*(s/v[-1])**b for s in v]

def interpolate_points_6(v, x, y):
    '''
    polynomial for point load
    '''
    a = y[1] / (3/2 * x[1]**2/v[-1]**2 - 1/2 * x[1]**3/v[-1]**3)
    return [a*(3/2 * s**2/v[-1]**2 - 1/2 * s**3/v[-1]**3) for s in v]

def interpolate_points_7(v, x, y):
    '''
    polynomial for uniform load
    '''
    a = y[1] / (2 * x[1]**2/v[-1]**2 - 4 * x[1]**3/v[-1]**3 + 1/3 * x[1]**4/v[-1]**4)
    return [a*(2 * s**2/v[-1]**2 - 4 * s**3/v[-1]**3 + 1/3 * s**4/v[-1]**4) for s in v]

# main definition of parameters

x = [0.0, 120.0]
y = [0.0, 120/250]
dydx = [0.0, y[1]/x[1]*2]
print(120/250)
print(np.rad2deg(np.arctan(y[1]/x[1]*2)))
s_x = np.linspace(x[0], 180, 100)

# plotting result

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(x,y, 'ko--')
# NOTE: why does this seem to curve in a different manner after 120.0?
plt.plot(s_x, interpolate_points_1(s_x, x, y, dydx), 'r--',label='1')
plt.plot(s_x, interpolate_points_2(s_x, x, y), 'k-',label='2')
plt.plot(s_x, interpolate_points_3(s_x, x, y, 1.5), 'b-..',label='3-1.5')
plt.plot(s_x, interpolate_points_3(s_x, x, y, 2), 'b-.',label='3-2')
# NOTE: why does this seem to curve in a different manner after 120.0?
plt.plot(s_x, interpolate_points_4(s_x, x, y), 'g-..',label='4')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(s_x, interpolate_points_2(s_x, x, y), 'k-',label='2')
plt.plot(s_x, interpolate_points_5(s_x, x, y, dydx), 'r--',label='5')
plt.plot(s_x, interpolate_points_6(s_x, x, y), 'b-..',label='6')
plt.plot(s_x, interpolate_points_7(s_x, x, y), 'b-.',label='7')
plt.legend()
plt.grid()

# NOTE: check estimatas from Chopra Table 5.8.1, sections 8.5.4, 8.6

plt.show()
