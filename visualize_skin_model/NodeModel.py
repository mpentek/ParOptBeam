from math import sin, cos
import numpy as np


class Node:
    def __init__(self, x, y, z):
        self.x0 = x
        self.y0 = y
        self.z0 = z
        self.x = self.x0
        self.y = self.y0
        self.z = self.z0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.theta_z = 0.0
    
    def assign_dofs(self, dx, dy, dz, theta_x, theta_y, theta_z):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z

    def print_info(self):
        msg = "###################################################\n"
        msg += "Node at Position [" + str(self.x0) + " "+ str(self.y0) + " " + str(self.z0) + "]: \n"
        try:
            msg += "dx = " + str(self.dx) + "\t"
            msg += "dy = " + str(self.dy) + "\t"
            msg += "dz = " + str(self.dz) + "\n"
            msg += "theta_x = " + str(self.theta_x) + "\t"
            msg += "theta_y = " + str(self.theta_y) + "\t"
            msg += "theta_z = " + str(self.theta_z)
        except AttributeError:
            pass
        print(msg)

    def apply_transformation(self):
        # translation matrix
        T = np.matrix([ [1,                  0,                    0,                   self.dx],
                        [0,                  1,                    0,                   self.dy],
                        [0,                  0,                    1,                   self.dz],
                        [0,                  0,                    0,                   1      ]])

        # rotation matrix around axis x
        Rx = np.matrix([[1,                  0,                    0,                   0],
                        [0,                  cos(self.theta_x),   -sin(self.theta_x),   0],
                        [0,                  sin(self.theta_x),    cos(self.theta_x),   0],
                        [0,                  0,                    0,                   1]])

        # rotation matrix around axis y
        Ry = np.matrix([[ cos(self.theta_y), 0,                    sin(self.theta_y),   0],
                        [0,                  1,                    0,                   0],
                        [-sin(self.theta_y), 0,                    cos(self.theta_y),   0],
                        [0,                  0,                    0,                   1]])

        # rotation matrix around axis z
        Rz = np.matrix([[cos(self.theta_z), -sin(self.theta_z),     0,                  0],
                        [sin(self.theta_z),  cos(self.theta_z),     0,                  0],
                        [0,                  0,                     1,                  0],
                        [0,                  0,                     0,                  1]])

        previous_coordinate = np.matrix([[self.x0],[self.y0],[self.z0],[1]])   
        new_coordinate = (T*Rz*Ry*Rx)*previous_coordinate
        self.x = float(new_coordinate[0][0])
        self.y = float(new_coordinate[1][0])
        self.z = float(new_coordinate[2][0])