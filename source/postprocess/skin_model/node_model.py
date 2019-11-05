from math import sin, cos
import numpy as np


class Node:
    def __init__(self, undeformed):
        self.undeformed = undeformed
        self.deformed = undeformed
        self.displacement = np.empty(3, dtype=float)
        self.angular_displacement = np.empty(3, dtype=float)

    def assign_dofs(self, displacement, angular_displacement):
        self.displacement = displacement
        self.angular_displacement = angular_displacement

    def print_info(self):
        msg = "########################################################\n"
        msg += "Node Fix Position [" + str(self.undeformed[0]) + " " + \
               str(self.undeformed[1]) + " " + \
            str(self.undeformed[2]) + "]: \n"
        try:
            msg += "dx = " + str(self.displacement[0]) + "\t"
            msg += "dy = " + str(self.displacement[1]) + "\t"
            msg += "dz = " + str(self.displacement[2]) + "\n"
            msg += "theta_x = " + str(self.angular_displacement[0]) + "\t"
            msg += "theta_y = " + str(self.angular_displacement[1]) + "\t"
            msg += "theta_z = " + str(self.angular_displacement[2]) + "\n"
            msg += "Transformed Position: [" + str(self.deformed[0]) + " " + \
                str(self.deformed[1]) + " " + str(self.deformed[2]) + "]:\n"
            msg += "########################################################\n"

        except AttributeError:
            pass
        print(msg)

    def apply_transformation(self):

        dx = self.displacement[0]
        dy = self.displacement[1]
        dz = self.displacement[2]

        alpha = self.angular_displacement[0]
        beta = self.angular_displacement[1]
        gamma = self.angular_displacement[2]

        # transformation matrix derived from the symbolic generation
        M = np.array([
            [1.0 * cos(beta) * cos(gamma),
             -1.0 * cos(alpha) * sin(gamma) + 1.0 *
             cos(gamma) * sin(alpha) * sin(beta),
             1.0 * cos(alpha) * cos(gamma) * sin(beta) + 1.0 * sin(alpha) * sin(gamma), 1.0 * dx],
            [1.0 * cos(beta) * sin(gamma), 1.0 * cos(alpha) * cos(gamma) + 1.0 * sin(alpha) * sin(beta) * sin(gamma),
             1.0 * cos(alpha) * sin(beta) * sin(gamma) - 1.0 * cos(gamma) * sin(alpha), 1.0 * dy],
            [-1.0 * sin(beta), 1.0 * cos(beta) * sin(alpha),
             1.0 * cos(alpha) * cos(beta), 1.0 * dz],
            [0, 0, 0, 1.0]
        ])

        previous_coordinate = np.append(
            self.undeformed, np.array(1.0)).reshape(4, 1)
        new_coordinate = M.dot(previous_coordinate)
        new_coordinate = new_coordinate.reshape(1, 4)
        self.deformed = new_coordinate[0][0:3]


def test():
    p = np.array([1.0, 1.0, 0.0])
    displacement = np.array([0.0, 0.0, 3.0])
    angular_displacement = np.array([0.0, 0.0, np.pi/2])
    node = Node(p)
    node.assign_dofs(displacement, angular_displacement)
    node.apply_transformation()
    node.print_info()
    solution = np.array([-1.0, 1.0, 3.0])
    err = solution - node.deformed

    try:
        assert all(err < 1e-12), "Transformation wrong"
        print("passed test")
    except AssertionError:
        print("failed test")


if __name__ == "__main__":
    test()
