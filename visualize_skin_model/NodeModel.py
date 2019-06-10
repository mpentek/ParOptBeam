from math import sin, cos
import numpy as np

class Node:
    def __init__(self, x, y, z):
        self.x0 = x
        self.y0 = y
        self.z0 = z
    
    def add_dofs(self, dx, dy, dz, theta_x, theta_y, theta_z):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z

    def print_info(self):
        msg = "###################################################\n"
        msg += "Node at Position [" + str(self.x0) + " "+ str(self.y0) + " " + str(self.z0) + "]: \n"
        msg += "dx = " + str(self.dx) + "\t" 
        msg += "dy = " + str(self.dy) + "\t" 
        msg += "dz = " + str(self.dz) + "\n" 
        msg += "theta_x = " + str(self.theta_x) + "\t" 
        msg += "theta_y = " + str(self.theta_y) + "\t" 
        msg += "theta_z = " + str(self.theta_z)
        print(msg)

    def apply_rotation(self):
        # rotation matrix around axis x
        Rx = np.matrix([[1,     0,                  0              ],
                        [0,     cos(self.theta_x), -sin(self.theta_x)],
                        [0,     sin(self.theta_x),  cos(self.theta_x) ]])

        # rotation matrix around axis y
        Ry = np.matrix([[ cos(self.theta_y),    0,    sin(self.theta_y)],
                        [0,                     1,    0              ],
                        [-sin(self.theta_y),    0,    cos(self.theta_y)]])

        # rotation matrix around axis z
        Rz = np.matrix([[cos(self.theta_z), -sin(self.theta_z),     0],
                        [sin(self.theta_z),  cos(self.theta_z),     0],
                        [0,                  0,                      1]])

        previous_coordinate = np.matrix([[self.x0],[self.y0],[self.z0]])
        new_coordinate = (Ry*Rx)*previous_coordinate
        self.x = float(new_coordinate[0][0])
        self.y = float(new_coordinate[1][0])
        self.z = float(new_coordinate[2][0])