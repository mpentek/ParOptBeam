from sympy import *

acos, asin = symbols('cos(alpha) sin(alpha)')
bcos, bsin = symbols('cos(beta) sin(beta)')
gcos, gsin = symbols('cos(gamma) sin(gamma)')

dx, dy, dz = symbols('dx dy dz')

# translation matrix
T = Matrix([[1.0, 0.0, 0.0, dx],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
            [0.0, 0.0, 0.0, 1.0]])

# rotation matrix around axis x
Rx = Matrix([[1.0, 0.0, 0.0, 0.0],
             [0.0, acos, -asin, 0.0],
             [0.0, asin, acos, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

# rotation matrix around axis y
Ry = Matrix([[bcos, 0, bsin, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [-bsin, 0.0, bcos, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

# rotation matrix around axis z
Rz = Matrix([[gcos, -gsin, 0.0, 0.0],
             [gsin, gcos, 0.0, 0.0],
             [0.0, 0.0, 1, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

M = T * Rz * Ry * Rx

if __name__ == "__main__":
    print(M)
