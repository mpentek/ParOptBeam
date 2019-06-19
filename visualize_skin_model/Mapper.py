from visualize_skin_model.NodeModel import Node
from visualize_skin_model.LineStructureModel import LineStructure
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np

# curvature - 2nd order deriv
DERIV_ORDER = 2
PRESCRIBED_2ND_ORDER_DERIV = [0, 0]
INTERPOLATION_DENSITY = 5


def extrap(v, x, y):
    if v <= x[0]:
        return y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
    elif v >= x[-1]:
        return y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
    else:
        f = interp1d(x, y, kind='cubic')
        return f(v)


def interpolate_points(x, f):
    interp_x = CubicSpline(x, f, bc_type=(
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[0]),
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[1])))
    return interp_x


class Mapper:
    def __init__(self, structure, line_structure):
        self.structure = structure
        self.line_structure = line_structure
        self.interpolated_line_structure = LineStructure()

    def map_line_structure_to_structure_floor(self):
        mid_z_prev = 0
        for floor in self.structure.floors:
            mid_x = sum(floor.x_vec) / len(floor.x_vec)
            mid_y = sum(floor.y_vec) / len(floor.y_vec)
            mid_z = sum(floor.z_vec) / len(floor.z_vec)

            self.map_to_interpolated_line_structure(mid_x, mid_y, mid_z, mid_z_prev)
            self.interpolate_dofs(mid_z, floor.nodes)

            mid_z_prev = mid_z

    def map_to_interpolated_line_structure(self, mid_x, mid_y, mid_z, mid_z_prev):
        z_vec = np.linspace(mid_z_prev, mid_z, INTERPOLATION_DENSITY)
        for z in z_vec:
            node = Node(mid_x, mid_y, z)
            self.interpolate_dofs(z, node)

            self.interpolated_line_structure.nodes.append(node)
            self.interpolated_line_structure.x_vec.append(node.x0)
            self.interpolated_line_structure.y_vec.append(node.y0)
            self.interpolated_line_structure.z_vec.append(node.z0)

    def interpolate_dofs(self, z, nodes):
        dx = extrap(z, self.line_structure.z_vec, self.line_structure.dx_vec)
        dy = extrap(z, self.line_structure.z_vec, self.line_structure.dy_vec)
        dz = extrap(z, self.line_structure.z_vec, self.line_structure.dz_vec)
        theta_x = extrap(z, self.line_structure.z_vec, self.line_structure.theta_x_vec)
        theta_y = extrap(z, self.line_structure.z_vec, self.line_structure.theta_y_vec)
        theta_z = extrap(z, self.line_structure.z_vec, self.line_structure.theta_z_vec)

        try:
            for node in nodes:
                node.add_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
        except:
            nodes.add_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
