from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

x = [0.5, 1.75, 2.65]
y = [0.32, 2.15, 9.12]

# curvature - 2nd order deriv
deriv_order = 2
# results linear interpolation
prescribed_2nd_order_deriv = [0, 0]
cs0 = CubicSpline(x, y, bc_type=(
    (deriv_order, prescribed_2nd_order_deriv[0]),
    (deriv_order, prescribed_2nd_order_deriv[1])))

# prescribe rotations = 1st oder deriv
deriv_order = 1
prescribed_1st_order_deriv = [3.3, 3.3]
cs1 = CubicSpline(x, y, bc_type=(
    (deriv_order, prescribed_1st_order_deriv[0]),
    (deriv_order, prescribed_1st_order_deriv[1])))

xs = np.linspace(x[0], x[-1], 100)

plt.figure()
plt.plot(x, y, 'o', label='endpoints')
plt.plot(xs, cs0(xs), label='cs0_lin_interp')
plt.plot(xs, cs1(xs), label='cs1_interp')

plt.legend(loc='best')
plt.show()
